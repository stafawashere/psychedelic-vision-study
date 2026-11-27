# What if my eyes move one pixel?

Normally I hand the model exactly the same aligned road each frame. That's a pretty generous setup for finding the road again. In the jitter test I shift the incoming photo by up to one pixel along each axis and add noise with sigma 2/255. I pad the edge before shifting so the model doesn't get a new black border to follow.

## Measure the shift itself first

Take two copies of the road. Move one a fraction of a pixel. A sharp edge can now sit BETWEEN pixels, so the interpolation changes the values on both sides. M-2 cares where those edges are. Even with the gain slope at zero the jitter probe produced pixel errors from 0.0383 to 0.0517. My ordinary return cutoff sits right in there at 0.05.

The actual jittered seed 0 ending scored 0.0318 RMSE and 0.686 detail. I keep that row beside the steady runs so I can see what changed. Giving it the same pass/fail interpretation would hide how much error comes from moving the input alone. I also only have one jittered seed.

> before blaming the loop for missing the road, see what moving the road does to the ruler.

[Jitter implementation](../../../run.py) · [Jitter-only floor](../../experiments/measurements.md)
