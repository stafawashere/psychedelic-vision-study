# What happens if I unwrap a ring?

Pick a center. Walk around it at the same radius. On a perfect ring I stay on the bright line the whole way. Now make angle the horizontal axis and distance from the center the vertical axis. The ring becomes a stripe across the page. A spoke through the center behaves the other way. Its angle stays fixed while its radius changes.

Log-polar uses log distance, so equal steps along that axis correspond to multiplying the radius. That's why a log spiral has such a simple shape once I unwrap it.

## I use this idea in two different places

In arm B I use a coordinate map on an activation while the UNet is working. That's me building a spatial assumption INTO the experiment. In M-1 I use coordinates to inspect the image AFTER the model finishes. Measuring arm A that way doesn't put a warp into its render.

The prediction for B was an increase in radial fraction or log-polar concentration. On the forest comparison radial fraction went from 0.038 in A to 0.023 in B. So having the right-looking coordinate map on paper didn't mean the final pixels organized around its center.

[Warp](../../../src/render/hooks.py) · [Arm A and B measurements](../../experiments/results.md)
