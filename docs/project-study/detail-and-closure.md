# Is the SAME road back?

Two pictures can have green at the top and brown at the bottom. They can both get captioned "dirt road." Move one branch a few pixels and I'd still call them the same place. For closure I'm asking something stricter. Did the little edges come back to the positions where they started?

RMSE checks the RGB values directly. Convert to floats in 0 to 1, subtract the first frame, square, average, take the square root. M-2 looks at the directions of the fine edges. Flip dark and light everywhere and the branch is still in the same place, so contrast inversion scores 1.000 on that control. Blur the branch away and detail falls to 0.052.

## What am I comparing against?

For return I use frame 00 OF THAT SEED. It's already been rendered once. Compare the actual photograph to frame 00 and I'm asking about damage from the first render too.

Seed 1 ended at 0.0329 pixel error and 0.813 detail. Both count. The limits are 0.05 and 0.8. It only had 0.013 to spare on detail. That's why I write "three seeds closed" and leave the fourth seed an open question.

[All three seed rows](experiments/results.md)
