# How does a sentence move pixels?

BLIP looks at the forest and gives me "a view of a dirt road in the middle of a forest." It does that once. I keep the sentence for the whole run and save it with the config. If the middle frame got captioned again I'd be letting something the model just drew change what I asked it to draw next.

## Classifier-free guidance

At a denoising step the model can predict noise with the caption and with empty conditioning. Both see the same noisy latent. Subtract their predictions and I get a direction. That's the part that changes when the words get a vote.

> direction = caption prediction - empty prediction  
> prediction I use = empty prediction + guidance scale times direction

So lowering prior precision from the upper setting of 12 toward 6 still leaves guidance on. And lowering it doesn't erase the model's learned idea of a tree. It changes how far I go along that difference.

Why filter the caption? If "spiral" gets into the text, the model already has a way to draw one before I test a single knob. The words in this experiment describe the input road. Any geometry has to come by another route.

[Caption filter and denoising loop](../../../src/render/engine.py)
