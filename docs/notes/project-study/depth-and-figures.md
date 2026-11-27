# Would turning up the gist make a figure?

In my UNet notes I called the bottleneck "the gist of the whole image." That gave me a prediction. Maybe poking shallow features gets me edges and textures, and poking deeper features gets me something with parts, maybe even a figure. There's a step in that reasoning I have to actually test. Storing information about an object and making an object appear when I multiply the whole block are different abilities.

## Which end is deep?

Block 0 is near the input side. Block 4 is the bottleneck. Block 8 is late in execution but back near full resolution. So picking 0 and 8 together because they're at opposite ends of the list doesn't mean I'm picking low-level AND high-level meaning. The list tells me execution order. The U shape tells me resolution.

The depth experiments didn't give me figures, including at the bottleneck. Multi-block profiles also made almost uniform images. The final forest config keeps depth pinned to 0. If a pattern gets more visible in that run, I can't explain it by saying the hook went deeper at peak. I can check both depth endpoints in the config and see they're the same.

That result limits this particular intervention. It can't tell me whether people see figures or how a human visual system builds one.

[Depth experiments](../../experiments/design.md) · [Fixed-depth config](../../../configs/deliverable_fall3.yaml)
