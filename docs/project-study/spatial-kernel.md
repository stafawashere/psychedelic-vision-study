# What would actually pick a wavelength?

Gain is one number. If two nearby cells are bright, gain can make both brighter. It doesn't contain a distance telling the next cell how far away to be. One idea was to give nearby cells one weight and farther cells another. A difference of Gaussians does this by subtracting a wide blur from a narrow one. Very slow changes show up in both blurs and cancel. Very fast changes get smoothed away. A band in the middle gets through.

## Cells or pixels?

My kernel widths live on a feature map. M-1 reports a spacing in IMAGE pixels. At block 0 one cell spans 16 image pixels. With the proposed minimum inner sigma of one cell, the smallest period the kernel could select was about 4.44 cells. Multiply that out and I get 71.1 pixels. M-1 stops at 64.

The late blocks look tempting. Eight pixels per cell there puts the floor at 35.5 px, inside the measurement window. Except hooking those blocks gave nearly flat frames. A wavelength on a blank frame can just be the instrument hitting its ceiling.

So E-3 closed at the arithmetic check. No kernel render was needed to find that mismatch. That's a different outcome from building a measurable kernel and watching its prediction fail.

[Kernel decision](experiments/design.md)
