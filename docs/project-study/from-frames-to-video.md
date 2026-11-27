# Which thing did I actually measure?

Open the movie and the forest comes and goes in a few seconds. In the config that same arc lasts twelve minutes. Where did the time go?

The model makes 40 numbered PNGs. The dose curve assigns each one a simulated time, 12/39 minutes apart. After the run I take those pictures in order and play them at five frames a second. I also hold the ends for playback. The extra time you see on a still image in the movie didn't give the loop extra frames to settle.

## Three different endings

- Last measured frame = 39.png. That's the one I compare to 00.png for closure.
- Last image in the video = another display of the ending. Compressing an MP4 can change pixel values, so I use the PNGs for measurements.
- Last dose value = a small positive number on the tail of the curve. It doesn't have to land exactly at zero for the image to get close to its start.

The GIF is handy for seeing the whole arc. The numbered frames tell me which part of that arc earned a score.

[Video assembly](../../src/measure/make_video.py) · [Seed 0 frames](../../results/seed-0/frames)
