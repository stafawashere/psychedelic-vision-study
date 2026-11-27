# How I got a second peak by accident

The original prior setting just goes down as dose goes up. E-6 tried a different shape. Let caption guidance get stronger at an intermediate dose, then let it weaken near the top. I set the turn at dose 0.5 and the guidance peak at 18.

Draw guidance against DOSE and there's one peak, right where I put it. Now draw it against FRAME. The dose passes 0.5 going up and passes 0.5 again coming down. The function sees the same input both times. It has no way to know I'm headed home the second time.

That second guidance peak reached 17.03 at frame 25. The caption still just says dirt road. Stronger guidance can push toward *a* road that fits the words while the little edges of MY road stop lining up. The comparison run closed at 0.0247 RMSE and 0.843 detail. With the bent guidance curve I got 0.0614 and 0.578. The early-detail test also got worse at seven of nine come-up frames.

> before I render anything, print the knob against frame number. plotting it against dose hides half the trip.

[Prior mapping](../../src/core/knobs.py) · [E-6 comparison](experiments/design.md)
