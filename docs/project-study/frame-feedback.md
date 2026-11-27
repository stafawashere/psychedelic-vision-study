# How frame 12 gets into frame 13

At the beginning prev = scene. There isn't a generated frame yet. After that I keep two versions of what just came out. The PNG I save is the decoded render. The one I feed forward gets its average color and contrast matched to the source photo first.

## Why two copies?

Say one frame came out a little pale. If I feed it right back in, the next frame starts with some of that paleness already there. Do it 40 times and a change in brightness can look like part of the dose curve. The feedback copy gets adjusted channel by channel. Matching the average green and its spread still can't tell the model where any leaf goes.

Next frame I encode that copy and the fresh photo separately, then blend their latents. Recurrence decides how much the previous one counts. It gives the loop a memory one frame long. Older stuff can survive if it keeps getting carried through those successive copies.

> saved frame = what came out. feedback copy = what the next frame gets to use.

[Follow the two copies in the runner](../../run.py)
