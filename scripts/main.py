from typing import List

import gradio as gr

import modules.scripts as scripts

from scripts.hook import (
    IllustriousPadEnhancerHook,
    AttentionWindowLimit,
    PositionalEncoding,
    AttentionScope,
)


class IllustriousPadEnhancer(scripts.Script):
    def __init__(self):
        super().__init__()
        self.hook = IllustriousPadEnhancerHook()

    def title(self) -> str:
        return "Illustrious Pad Enhancer"

    def show(self, is_img2img: bool) -> object:
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> List[gr.components.Component]:
        with gr.Accordion("Illustrious Pad Enhancer", open=False):
            pad_enable = gr.Checkbox(value=False, label="Enable Pad Enhancer")
            pad_enable_for = gr.CheckboxGroup(
                choices=["Positive Prompt", "Negative Prompt"],
                value=["Positive Prompt"],
                label="Enable for",
            )
            pad_length = gr.Slider(
                minimum=1, maximum=75, step=1, label="Pad Length", value=15
            )
            pad_attention_scope = gr.CheckboxGroup(
                choices=[t.value for t in AttentionScope],
                value=[
                    AttentionScope.BOS.value,
                    AttentionScope.PROMPT.value,
                    AttentionScope.EOS.value,
                    AttentionScope.ADDITIONAL_PAD.value,
                ],
                label="Attention Scope",
            )
            pad_attention_window_limit = gr.Radio(
                choices=[t.value for t in AttentionWindowLimit],
                value=AttentionWindowLimit.UNLIMITED.value,
                label="Attention Window Limit",
            )
            pad_positional_encoding = gr.Radio(
                choices=[t.value for t in PositionalEncoding],
                value=PositionalEncoding.FROM_END.value,
                label="Positional Encoding",
            )
            pad_enable_normalization = gr.Checkbox(
                value=False, label="Enable Normalization"
            )
            pad_to_additional_pad_attention = gr.Checkbox(
                value=True, label="Pad to Additional Pad Attention"
            )
        return [
            pad_enable,
            pad_enable_for,
            pad_length,
            pad_attention_scope,
            pad_attention_window_limit,
            pad_positional_encoding,
            pad_enable_normalization,
            pad_to_additional_pad_attention,
        ]

    def process(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.hook.process(*args, **kwargs)

    def process_before_every_sampling(self, *args, **kwargs) -> None:
        self.hook.process_before_every_sampling(*args, **kwargs)
