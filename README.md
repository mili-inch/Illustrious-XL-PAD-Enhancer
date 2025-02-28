# Illustrious-XL-PAD-Enhancer

## Description

In image generation models, particularly in Illustrious-XL and its variants, I had the impression that the generated results might degrade when using prompts that reach the CLIP Text Encoder's maximum length of 77 tokens. This often results in more unnatural artifacts and a blurring of fine details. However, I am not entirely confident about this observation.

I hypothesize that this could be due to the CLIP weights not being frozen during the model's fine-tuning process, leading the padding tokens to learn to extract information from other prompt tokens or to encode information necessary for denoising. This may lead to issues when there are not enough padding tokens.

This extension ensures that additional padding tokens are added to the CLIP Text Encoder input when the number of padding tokens is low.

## Installation

```bash
cd extensions
git clone https://github.com/mili-inch/Illustrious-XL-PAD-Enhancer
```

## Options

- PAD Length: The length of the extra padding tokens to add.
- PAD Enable For: The option to enable the extra padding tokens for the positive or negative prompt.
- PAD Enable Normalization: Whether to normalize the extra padding tokens.

## Credits

- [Onoma AI](https://onomaai.com/) - Thank you for publicly releasing the models!
