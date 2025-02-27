from typing import List, Optional, Callable
from enum import Enum
import torch

from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings,
)
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

from modules.processing import StableDiffusionProcessing
from modules.shared import opts as shared_opts
from modules.sd_emphasis import get_current_option


class AttentionScope(Enum):
    BOS = "BOS"
    PROMPT = "Prompt"
    EOS = "EOS"
    PAD = "PAD"
    ADDITIONAL_PAD = "Additional PAD"


class AttentionWindowLimit(Enum):
    UNLIMITED = "Unlimited"
    POSITION_BASED_FROM_END = "PositionBasedFromEnd"
    POSITION_BASED_FROM_START = "PositionBasedFromStart"


class PositionalEncoding(Enum):
    FROM_END = "FromEnd"
    FROM_START = "FromStart"
    UNUSED_FROM_END = "UnusedFromEnd"
    UNUSED_FROM_START = "UnusedFromStart"
    LAST_REPEAT = "LastRepeat"
    NO_PE = "NoPE"
    MEAN_REPEAT = "MeanRepeat"


class IllustriousPadEnhancerHook:
    def __init__(self):
        self.original_encoder_forward_l: Optional[object] = None
        self.original_encoder_forward_g: Optional[object] = None
        self.original_embeddings_forward_l: Optional[object] = None
        self.original_embeddings_forward_g: Optional[object] = None
        self.eos_position: Optional[int] = None
        self.previous_options: Optional[dict] = None

    def _get_patched_clip_encoder_forward(
        self,
        original_encoder_forward: Callable,
        pad_attention_scope: List[AttentionScope],
        pad_attention_window_limit: AttentionWindowLimit,
        pad_positional_encoding: PositionalEncoding,
    ):
        def forward(
            inputs_embeds,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        ):
            new_causal_attention_mask = _create_4d_causal_attention_mask(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=causal_attention_mask.dtype,
                device=causal_attention_mask.device,
            )

            # get mask
            mask = new_causal_attention_mask[
                0, 0, 0, inputs_embeds.shape[1] - 1
            ].clone()

            # apply mask to the attention mask for each scope
            if AttentionScope.BOS.value not in pad_attention_scope:
                new_causal_attention_mask[:, :, 77:, 0] = mask
            if AttentionScope.PROMPT.value not in pad_attention_scope:
                new_causal_attention_mask[:, :, 77:, 1 : self.eos_position] = mask
            if AttentionScope.EOS.value not in pad_attention_scope:
                new_causal_attention_mask[:, :, 77:, self.eos_position] = mask
            if AttentionScope.PAD.value not in pad_attention_scope:
                new_causal_attention_mask[:, :, 77:, self.eos_position + 1 : 77] = mask
            if AttentionScope.ADDITIONAL_PAD.value not in pad_attention_scope:
                new_causal_attention_mask[:, :, 77:, 77:] = mask

            # apply attention window limit
            additional_pad_length = inputs_embeds.shape[1] - 77
            additional_pad_positions = []
            if PositionalEncoding.FROM_START.value in pad_positional_encoding:
                for i in range(additional_pad_length):
                    additional_pad_positions.append(1 + i)
            elif PositionalEncoding.FROM_END.value in pad_positional_encoding:
                for i in range(additional_pad_length):
                    additional_pad_positions.append(77 - additional_pad_length + i + 1)
            elif PositionalEncoding.LAST_REPEAT.value in pad_positional_encoding:
                for i in range(additional_pad_length):
                    additional_pad_positions.append(77)

            if (
                pad_attention_window_limit
                == AttentionWindowLimit.POSITION_BASED_FROM_START.value
            ):
                for i in range(additional_pad_length):
                    not_masked_count = 0
                    for j in range(inputs_embeds.shape[1]):
                        if not_masked_count > additional_pad_positions[i]:
                            new_causal_attention_mask[:, :, 77 + i, j] = mask
                            continue
                        if new_causal_attention_mask[:, :, 77 + i, j] >= 0:
                            not_masked_count += 1
            elif (
                pad_attention_window_limit
                == AttentionWindowLimit.POSITION_BASED_FROM_END.value
            ):
                for i in range(additional_pad_length):
                    not_masked_count = 0
                    for j in reversed(range(inputs_embeds.shape[1])):
                        if not_masked_count > additional_pad_positions[i]:
                            new_causal_attention_mask[:, :, 77 + i, j] = mask
                            continue
                        if new_causal_attention_mask[:, :, 77 + i, j] >= 0:
                            not_masked_count += 1

            # diagonal
            for i in range(additional_pad_length):
                new_causal_attention_mask[:, :, 77 + i, 77 + i] = 0

            return original_encoder_forward(
                inputs_embeds,
                attention_mask,
                new_causal_attention_mask,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        return forward

    def _get_patched_clip_text_embeddings_forward(
        self,
        embeddings: CLIPTextEmbeddings,
        original_embeddings_forward: Callable,
        id_end: int,
        id_pad: int,
        pad_length: int,
        pad_positional_encoding: PositionalEncoding,
    ):
        pad_embed = embeddings.token_embedding.weight[id_pad]  # D
        position_embeddings = embeddings.position_embedding.weight
        mean_position_embedding = position_embeddings.data.mean(dim=0)  # D

        def forward(
            input_ids=None,  # B, L
            position_ids=None,  # None
            inputs_embeds=None,  # None
        ):
            self.eos_position = (input_ids == id_end).nonzero()[0, 1].item()
            original_embeddings = original_embeddings_forward(
                input_ids, position_ids, inputs_embeds
            )  # B, L, D
            additional_embeddings = pad_embed.repeat(
                original_embeddings.shape[0], pad_length, 1
            )
            additional_position_embeddings = None
            if pad_positional_encoding == PositionalEncoding.FROM_START.value:
                additional_position_embeddings = position_embeddings[1 : pad_length + 1]
            elif pad_positional_encoding == PositionalEncoding.FROM_END.value:
                additional_position_embeddings = position_embeddings[-pad_length:]
            elif pad_positional_encoding == PositionalEncoding.NO_PE.value:
                additional_position_embeddings = torch.zeros(
                    pad_length,
                    position_embeddings.shape[1],
                    device=original_embeddings.device,
                )
            elif pad_positional_encoding == PositionalEncoding.LAST_REPEAT.value:
                additional_position_embeddings = position_embeddings[-1].repeat(
                    pad_length, 1
                )
            elif pad_positional_encoding == PositionalEncoding.MEAN_REPEAT.value:
                additional_position_embeddings = mean_position_embedding.repeat(
                    pad_length, 1
                )
            else:
                raise ValueError(
                    f"Invalid positional encoding: {pad_positional_encoding}"
                )

            extended_embeddings = torch.cat(
                [
                    original_embeddings,
                    additional_embeddings.to("cuda")
                    + additional_position_embeddings.to("cuda"),
                ],
                dim=1,
            )
            return extended_embeddings

        return forward

    def _patch_emphasis(self):
        emphasis = get_current_option(shared_opts.emphasis)

        def after_transformers(self):
            if self.z.shape[1] > 77:
                z_temp = self.z[:, 77:, :]
                self.z = self.z[:, :77, :]
                original_after_transformers(self)
                self.z = torch.cat([self.z, z_temp], dim=1)
            else:
                original_after_transformers(self)

        original_after_transformers = emphasis.after_transformers
        emphasis.after_transformers = after_transformers

    def process(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        pad_length: int,
        pad_attention_scope: List[AttentionScope],
        pad_attention_window_limit: AttentionWindowLimit,
        pad_positional_encoding: PositionalEncoding,
        pad_enable_normalization: bool,
    ) -> None:
        # clear prompt cache
        current_options = {
            "pad_enable": pad_enable,
            "pad_enable_for": pad_enable_for,
            "pad_length": pad_length,
            "pad_attention_scope": pad_attention_scope,
            "pad_attention_window_limit": pad_attention_window_limit,
            "pad_positional_encoding": pad_positional_encoding,
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
            return

        if self.original_encoder_forward_l is None:
            self.original_encoder_forward_l = text_model_l.encoder.forward
        if self.original_encoder_forward_g is None:
            self.original_encoder_forward_g = text_model_g.encoder.forward
        if self.original_embeddings_forward_l is None:
            self.original_embeddings_forward_l = text_model_l.embeddings.forward
        if self.original_embeddings_forward_g is None:
            self.original_embeddings_forward_g = text_model_g.embeddings.forward

        self._patch_emphasis()

        text_model_l.encoder.forward = self._get_patched_clip_encoder_forward(
            self.original_encoder_forward_l,
            pad_attention_scope,
            pad_attention_window_limit,
            pad_positional_encoding,
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
                pad_positional_encoding,
            )
        )
        text_model_g.encoder.forward = self._get_patched_clip_encoder_forward(
            self.original_encoder_forward_g,
            pad_attention_scope,
            pad_attention_window_limit,
            pad_positional_encoding,
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
                pad_positional_encoding,
            )
        )

    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        pad_enable: bool,
        pad_enable_for: List[str],
        pad_length: int,
        pad_attention_scope: List[AttentionScope],
        pad_attention_window_limit: AttentionWindowLimit,
        pad_positional_encoding: PositionalEncoding,
        pad_enable_normalization: bool,
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

            # normalize conditioning using the mean of the original conditioning
            if "Positive Prompt" in pad_enable_for and pad_enable_normalization:
                for i in range(len(cond)):
                    original_mean = (
                        cond[i]["model_conds"]["c_crossattn"].cond[:, :77, :].mean()
                    )
                    current_mean = cond[i]["model_conds"]["c_crossattn"].cond.mean()
                    cond[i]["model_conds"]["c_crossattn"].cond = cond[i]["model_conds"][
                        "c_crossattn"
                    ].cond * (original_mean / current_mean)

            if "Negative Prompt" in pad_enable_for and pad_enable_normalization:
                for i in range(len(uncond)):
                    original_mean = (
                        uncond[i]["model_conds"]["c_crossattn"].cond[:, :77, :].mean()
                    )
                    current_mean = uncond[i]["model_conds"]["c_crossattn"].cond.mean()
                    uncond[i]["model_conds"]["c_crossattn"].cond = uncond[i][
                        "model_conds"
                    ]["c_crossattn"].cond * (original_mean / current_mean)

            return model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet = p.sd_model.forge_objects.unet
        unet.add_conditioning_modifier(conditioning_modifier)

        p.extra_generation_params.update(self.previous_options)

        return
