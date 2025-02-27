from typing import List

import gradio as gr

import modules.scripts as scripts

from scripts.hook import (
    IllustriousPadEnhancerHook,
)


class IllustriousPadEnhancer(scripts.Script):
    def __init__(self):
        super().__init__()
        self.hook = IllustriousPadEnhancerHook()

    def title(self) -> str:
        return "Illustrious XL Pad Enhancer"

    def show(self, is_img2img: bool) -> object:
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> List[gr.components.Component]:
        with gr.Accordion("Illustrious XL Pad Enhancer", open=False):
            pad_enable = gr.Checkbox(value=False, label="Enable Pad Enhancer")
            pad_enable_for = gr.CheckboxGroup(
                choices=["Positive Prompt", "Negative Prompt"],
                value=["Positive Prompt"],
                label="Enable for",
            )
            pad_length = gr.Slider(
                minimum=1, maximum=75, step=1, label="Pad Length", value=15
            )
            pad_enable_normalization = gr.Checkbox(
                value=False, label="Enable Normalization"
            )
        return [
            pad_enable,
            pad_enable_for,
            pad_length,
            pad_enable_normalization,
        ]

    def process(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.hook.process(*args, **kwargs)

    def process_before_every_sampling(self, *args, **kwargs) -> None:
        self.hook.process_before_every_sampling(*args, **kwargs)
