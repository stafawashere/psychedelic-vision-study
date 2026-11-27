# Where are the eyes in this model?

Look at a branch hanging over the dirt road. The caption only says there's a forest. It doesn't say a branch bends left at THIS spot. That bend gets into the render through the photo. That's what I mean by sensory evidence here.

The model also has expectations built into its weights. It already knows a lot of ways a forest could look. The caption gives it a direction to use that knowledge. Turning down caption guidance lets that direction count for less. All the learned weights are still there.

## Brain relationship

REBUS gave me the idea of loosening a high-level expectation. The switch I actually have access to is a number in the denoiser. I can see what lowering that number does to the branch. I don't have a conversion from guidance scale to how strongly a person believes what they're seeing.

Same thing with cortical gain. I can multiply a layer's activations. A layer getting louder is the piece of the idea I'm borrowing. There's no receptor in the UNet that I can turn on.

> photo = where this branch actually is. caption = roughly what kind of scene belongs here.

[The question and its sources](../../paper/manuscript.md)
