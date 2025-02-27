from typing import List, Optional, Callable
import torch

from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings,
)
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

from modules.processing import StableDiffusionProcessing
from modules.shared import opts as shared_opts
from modules.sd_emphasis import get_current_option


class IllustriousPadEnhancerHook:
    def __init__(self):
        self.original_encoder_forward_l: Optional[object] = None
        self.original_encoder_forward_g: Optional[object] = None
        self.original_embeddings_forward_l: Optional[object] = None
        self.original_embeddings_forward_g: Optional[object] = None
        self.original_after_transformers: Optional[object] = None
        self.eos_position: Optional[int] = None
        self.previous_options: Optional[dict] = None
        self.conds: List[dict] = []

    def _get_patched_clip_encoder_forward(
        self,
        original_encoder_forward: Callable,
    ):
        def forward(
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        ):
            # create new causal attention mask
            new_causal_attention_mask = _create_4d_causal_attention_mask(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=causal_attention_mask.dtype,
                device=causal_attention_mask.device,
            )

            # get mask score
            mask = new_causal_attention_mask[
                0, 0, 0, inputs_embeds.shape[1] - 1
            ].clone()

            # additional pads must not attend to existing pads
            new_causal_attention_mask[:, :, 77:, self.eos_position + 1 : 77] = mask

            encoder_outputs = original_encoder_forward(
                inputs_embeds,
                attention_mask,
                new_causal_attention_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
            )
            print(encoder_outputs.hidden_states)
            return encoder_outputs

        return forward

    def _get_patched_clip_text_embeddings_forward(
        self,
        embeddings: CLIPTextEmbeddings,
        original_embeddings_forward: Callable,
        id_end: int,
        id_pad: int,
        pad_length: int,
    ):
        pad_embed = embeddings.token_embedding.weight[id_pad]  # D
        position_embeddings = embeddings.position_embedding.weight

        def forward(
            input_ids=None,  # B, L
            position_ids=None,
            inputs_embeds=None,
        ):
            # save eos position for later use
            self.eos_position = (input_ids == id_end).nonzero()[0, 1].item()

            # get original embeddings
            original_embeddings = original_embeddings_forward(
                input_ids, position_ids, inputs_embeds
            )  # B, L, D

            # create additional embeddings
            additional_embeddings = pad_embed.repeat(
                original_embeddings.shape[0], pad_length, 1
            ).to("cuda")
            additional_position_embeddings = position_embeddings[-pad_length:].to(
                "cuda"
            )

            # extend embeddings
            extended_embeddings = torch.cat(
                [
                    original_embeddings,
                    additional_embeddings + additional_position_embeddings,
                ],
                dim=1,
            )
            return extended_embeddings

        return forward

    def _get_patched_emphasis_after_transformers(
        self,
        original_after_transformers: Callable,
    ):
        def after_transformers(self):
            if self.z.shape[1] > 77:
                z_temp = self.z[:, 77:, :]
                self.z = self.z[:, :77, :]
                original_after_transformers(self)
                self.z = torch.cat([self.z, z_temp], dim=1)
            else:
                original_after_transformers(self)

        return after_transformers

    def process(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        pad_length: int,
        pad_enable_normalization: bool,
    ) -> None:
        # clear prompt cache
        current_options = {
            "pad_enable": pad_enable,
            "pad_enable_for": pad_enable_for,
            "pad_length": pad_length,
            "pad_enable_normalization": pad_enable_normalization,
        }

        if (
            self.previous_options is not None
            and self.previous_options != current_options
        ):
            p.cached_c = [None, None, None]
            p.cached_uc = [None, None, None]

        self.previous_options = current_options

        if pad_enable:
            p.cached_c = [None, None, None]
            p.cached_uc = [None, None, None]

        # check if the model is SDXL
        if not hasattr(
            p.sd_model.forge_objects.clip.cond_stage_model, "clip_l"
        ) or not hasattr(p.sd_model.forge_objects.clip.cond_stage_model, "clip_g"):
            return

        # patch CLIP text encoders
        text_model_l = (
            p.sd_model.forge_objects.clip.cond_stage_model.clip_l.transformer.text_model
        )
        text_model_g = (
            p.sd_model.forge_objects.clip.cond_stage_model.clip_g.transformer.text_model
        )
        emphasis = get_current_option(shared_opts.emphasis)

        if not pad_enable:
            if self.original_encoder_forward_l is not None:
                text_model_l.encoder.forward = self.original_encoder_forward_l
                self.original_encoder_forward_l = None
            if self.original_encoder_forward_g is not None:
                text_model_g.encoder.forward = self.original_encoder_forward_g
                self.original_encoder_forward_g = None
            if self.original_embeddings_forward_l is not None:
                text_model_l.embeddings.forward = self.original_embeddings_forward_l
                self.original_embeddings_forward_l = None
            if self.original_embeddings_forward_g is not None:
                text_model_g.embeddings.forward = self.original_embeddings_forward_g
                self.original_embeddings_forward_g = None
            if self.original_after_transformers is not None:
                emphasis.after_transformers = self.original_after_transformers
                self.original_after_transformers = None
            return

        if self.original_encoder_forward_l is None:
            self.original_encoder_forward_l = text_model_l.encoder.forward
        if self.original_encoder_forward_g is None:
            self.original_encoder_forward_g = text_model_g.encoder.forward
        if self.original_embeddings_forward_l is None:
            self.original_embeddings_forward_l = text_model_l.embeddings.forward
        if self.original_embeddings_forward_g is None:
            self.original_embeddings_forward_g = text_model_g.embeddings.forward
        if self.original_after_transformers is None:
            self.original_after_transformers = emphasis.after_transformers

        text_model_l.encoder.forward = self._get_patched_clip_encoder_forward(
            self.original_encoder_forward_l,
        )
        text_model_l.embeddings.forward = (
            self._get_patched_clip_text_embeddings_forward(
                text_model_l.embeddings,
                self.original_embeddings_forward_l,
                p.sd_model.forge_objects.clip.cond_stage_model.clip_l.special_tokens[
                    "end"
                ],
                p.sd_model.forge_objects.clip.cond_stage_model.clip_l.special_tokens[
                    "pad"
                ],
                pad_length,
            )
        )
        text_model_g.encoder.forward = self._get_patched_clip_encoder_forward(
            self.original_encoder_forward_g,
        )
        text_model_g.embeddings.forward = (
            self._get_patched_clip_text_embeddings_forward(
                text_model_g.embeddings,
                self.original_embeddings_forward_g,
                p.sd_model.forge_objects.clip.cond_stage_model.clip_g.special_tokens[
                    "end"
                ],
                p.sd_model.forge_objects.clip.cond_stage_model.clip_g.special_tokens[
                    "pad"
                ],
                pad_length,
            )
        )
        emphasis.after_transformers = self._get_patched_emphasis_after_transformers(
            self.original_after_transformers
        )

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        *args,
        **kwargs,
    ) -> None:
        if not pad_enable:
            return

        def conditioning_modifier(
            model, x, timestep, uncond, cond, cond_scale, model_options, seed
        ):
            # remove additional PAD if not enabled
            if not "Positive Prompt" in pad_enable_for:
                for i in range(len(cond)):
                    cond[i]["model_conds"]["c_crossattn"].cond = cond[i]["model_conds"][
                        "c_crossattn"
                    ].cond[:, :77, :]
            if not "Negative Prompt" in pad_enable_for:
                for i in range(len(uncond)):
                    uncond[i]["model_conds"]["c_crossattn"].cond = uncond[i][
                        "model_conds"
                    ]["c_crossattn"].cond[:, :77, :]
            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet = p.sd_model.forge_objects.unet
        unet.add_conditioning_modifier(conditioning_modifier)

        p.extra_generation_params.update(self.previous_options)

        return
