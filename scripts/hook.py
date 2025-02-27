from typing import List, Optional, Callable
import torch

from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings,
)
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

from modules.processing import StableDiffusionProcessing


class IllustriousPadEnhancerHook:
    def __init__(self):
        self.original_encoder_forward_l: Optional[object] = None
        self.original_encoder_forward_g: Optional[object] = None
        self.original_embeddings_forward_l: Optional[object] = None
        self.original_embeddings_forward_g: Optional[object] = None
        self.original_encode_with_transformers_l: Optional[object] = None
        self.original_encode_with_transformers_g: Optional[object] = None
        self.eos_position: Optional[int] = None
        self.previous_options: Optional[dict] = None
        self.conds: List[dict] = []

    def _get_patched_clip_encode_with_transformers(
        self,
        original_encode_with_transformers: Callable,
    ):
        def encode_with_transformers(
            tokens,
        ):
            output = original_encode_with_transformers(tokens)
            truncated_output = output[:, :77, :]
            extra_output = output[:, 77:, :]
            self.conds.append(
                {
                    "extra_output": extra_output,
                    "eos_position": self.eos_position,
                }
            )
            if hasattr(output, "pooled"):
                truncated_output.pooled = output.pooled

            return truncated_output

        return encode_with_transformers

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

    def process(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        pad_length: int,
        pad_enable_normalization: bool,
    ) -> None:
        # clear conds
        self.conds = []
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
        text_model_l = p.sd_model.clip.embedders[0].wrapped.transformer.text_model
        text_model_g = p.sd_model.clip.embedders[1].wrapped.transformer.text_model

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
            if self.original_encode_with_transformers_l is not None:
                p.sd_model.clip.embedders[
                    0
                ].encode_with_transformers = self.original_encode_with_transformers_l
                self.original_encode_with_transformers_l = None
            if self.original_encode_with_transformers_g is not None:
                p.sd_model.clip.embedders[
                    1
                ].encode_with_transformers = self.original_encode_with_transformers_g
                self.original_encode_with_transformers_g = None
            return

        if self.original_encoder_forward_l is None:
            self.original_encoder_forward_l = text_model_l.encoder.forward
        if self.original_encoder_forward_g is None:
            self.original_encoder_forward_g = text_model_g.encoder.forward
        if self.original_embeddings_forward_l is None:
            self.original_embeddings_forward_l = text_model_l.embeddings.forward
        if self.original_embeddings_forward_g is None:
            self.original_embeddings_forward_g = text_model_g.embeddings.forward
        if self.original_encode_with_transformers_l is None:
            self.original_encode_with_transformers_l = p.sd_model.clip.embedders[
                0
            ].encode_with_transformers
        if self.original_encode_with_transformers_g is None:
            self.original_encode_with_transformers_g = p.sd_model.clip.embedders[
                1
            ].encode_with_transformers

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
        p.sd_model.clip.embedders[
            0
        ].encode_with_transformers = self._get_patched_clip_encode_with_transformers(
            self.original_encode_with_transformers_l,
        )
        p.sd_model.clip.embedders[
            1
        ].encode_with_transformers = self._get_patched_clip_encode_with_transformers(
            self.original_encode_with_transformers_g,
        )

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        pad_length: int,
        pad_enable_normalization: bool,
        *args,
        **kwargs,
    ) -> None:
        if not pad_enable:
            return

        def conditioning_modifier(
            model, x, timestep, uncond, cond, cond_scale, model_options, seed
        ):
            extra_conds = []
            dimension = None
            for i in range(len(self.conds)):
                current_dimension = self.conds[i]["extra_output"].shape[2]
                if dimension is None or dimension != current_dimension:
                    dimension = current_dimension
                    extra_conds.append([])
                extra_conds[-1].append(self.conds[i])

            if "Negative Prompt" in pad_enable_for:
                split_uncond = torch.split(
                    uncond[0]["model_conds"]["c_crossattn"].cond, 77, dim=1
                )
                uncond_results = []
                for j in range(len(split_uncond)):
                    extra_uncond_l = extra_conds[0][j]
                    extra_uncond_g = extra_conds[1][j]
                    eos_position_l = extra_uncond_l["eos_position"]
                    eos_position_g = extra_uncond_g["eos_position"]

                    original_pad_length = 77 - eos_position_l - 1

                    if original_pad_length > pad_length:
                        uncond_results.append(split_uncond[j])
                        continue

                    split_uncond_l, split_uncond_g = torch.split(
                        split_uncond[j], [768, 1280], dim=2
                    )

                    extended_uncond_l = torch.cat(
                        [
                            split_uncond_l[:, : eos_position_l + 1, :],
                            extra_uncond_l["extra_output"],
                        ],
                        dim=1,
                    )
                    extended_uncond_g = torch.cat(
                        [
                            split_uncond_g[:, : eos_position_g + 1, :],
                            extra_uncond_g["extra_output"],
                        ],
                        dim=1,
                    )
                    uncond_results.append(
                        torch.cat([extended_uncond_l, extended_uncond_g], dim=2)
                    )
                uncond[0]["model_conds"]["c_crossattn"].cond = torch.cat(
                    uncond_results, dim=1
                )

            if "Positive Prompt" in pad_enable_for:
                split_cond = torch.split(
                    cond[0]["model_conds"]["c_crossattn"].cond, 77, dim=1
                )
                cond_results = []
                for j in range(len(split_cond)):
                    extra_cond_l = extra_conds[2][j]
                    extra_cond_g = extra_conds[3][j]
                    eos_position_l = extra_cond_l["eos_position"]
                    eos_position_g = extra_cond_g["eos_position"]

                    original_pad_length = 77 - eos_position_l - 1

                    if original_pad_length > pad_length:
                        cond_results.append(split_cond[j])
                        continue

                    split_cond_l, split_cond_g = torch.split(
                        split_cond[j], [768, 1280], dim=2
                    )

                    extended_cond_l = torch.cat(
                        [
                            split_cond_l[:, : eos_position_l + 1, :],
                            extra_cond_l["extra_output"],
                        ],
                        dim=1,
                    )
                    extended_cond_g = torch.cat(
                        [
                            split_cond_g[:, : eos_position_g + 1, :],
                            extra_cond_g["extra_output"],
                        ],
                        dim=1,
                    )
                    cond_results.append(
                        torch.cat([extended_cond_l, extended_cond_g], dim=2)
                    )
                cond[0]["model_conds"]["c_crossattn"].cond = torch.cat(
                    cond_results, dim=1
                )

            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet = p.sd_model.forge_objects.unet
        unet.add_conditioning_modifier(conditioning_modifier)

        p.extra_generation_params.update(self.previous_options)

        return
