# How I tested what happens between the road leaving and coming back

The first video already did the easy-to-describe part. I could point to the road at the start, lose it at peak and find it again at the end. What I couldn't point to was a lattice that lined up with itself, or a pattern that kept moving once I held the dose still. That's what this plan was for.

I wrote the decisions down as I went. Some things that were next on the list in the morning had already failed by the end of the day. This version follows the questions in the order I could answer them. I keep the original numbered protocol in the source project. Here I use the final decisions and link the measurements that settled them.

## What would count as the whole thing?

One config has to make the whole arc. If I take the peak from one run and the return from another, I've described a video my model never made.

| Part of the arc | What I decided to measure | What happened |
|---|---|---|
| Come-up | Fine-detail match (M-2) above 0.6 through frame 11. The plan left the reference ambiguous, so I checked the starting render and the actual photo. | Detail fell too early against either reference. |
| Geometry | Peak spectral bump above 5, with mean and SD over at least three seeds or runs. Record the measured spacing alongside a known lattice. | Mean 1.80, sample SD 0.85 at three seeds. |
| Symmetry | Mirror or 60-degree rotation above 0.5. For radial patterns in arm B, radial fraction above 0.5 or log-polar concentration above 385. Compare the gap to the seed spread. | Neither arm produced the required order. |
| Motion | With the knobs held still, change keeps going, consecutive patterns line up above 0.7 correlation, and the result separates from both settling and random churn. | Ordinary noise settled. Moving the noise increased change but cost the full arc its return. |
| Return | Compare the last render to its own frame 00. RMSE at most 0.05 and M-2 at least 0.8. Include a different photo and a separate jitter row. | All three steady-input seeds closed. Jitter alone can cross the pixel-error limit. |
| Integrity | Run the config and implementation checks, keep every seed, and label which arm produced it. | The reporting session passed verification. |

Arm A is the one where the model has to make the pattern with no spatial warp. Arm B gives it a log-polar map. I can learn something from either one. I have to keep track of which part of the geometry I put in myself.

## First build a score that can tell when I'm wrong

I used a hex lattice, rings and a spiral to check what the structure score sees. Blur the scene and the fine-detail score should drop. Invert its contrast and the edges are still in the right places, so it should stay high. Shuffle frames into churn and the motion score should catch that too. That's phase 0. Once the controls separated, I could spend renders on the actual questions. [Metric definitions and controls](measurements.md)

## Can I keep the branches sharp for longer?

**E-1. Hold denoise strength flat.** More noise means the photo starts each frame with less fine structure to work from. I tried letting gain carry the come-up with denoise strength fixed at 0.45. At frame 10 detail rose from 0.144 to 0.432 in this comparison. Then I checked the peak. Its periodic excess fell to 1.653 times the background, below the photo's own 1.873. Helping the branches cost me the peak.

**E-2. Move the input a little.** The ordinary run gets the same aligned photo every time. I jittered it by up to one pixel and added noise at sigma 2/255. At rest, that alone gave 0.0383 to 0.0517 pixel error. My return limit is 0.05. I kept the result as a separate robustness row. Next time I'd measure the perturbation by itself before deciding what the comparison can tell me.

## Which piece of the UNet should get louder?

**E-4. Follow the skip.** Down blocks hand back a forward summary and copies that travel across the UNet. I tried amplifying the copies. The frame went nearly blank, image SD 0.0028 against 0.1574 on the forward path. Amplifying both paths looked a little better at seed 0. Across seeds the average bumps were 2.13 and 2.12. There was no reason to spend a whole arc on that difference.

**E-5. Poke several levels together.** Every tested profile went nearly uniform, including blocks 0, 4 and 8 together. Block 8 is close to the noise prediction itself. Saturate it there and there's hardly any network left to work on what comes out. That's my explanation for the collapse. The observed result was the blank frame, and I closed the multi-block route.

**E-6. Bend the caption-guidance curve.** I put a turn in the mapping at dose 0.5. Look at it against *time* and the mistake gets obvious. The dose passes through 0.5 twice. I got another guidance peak of 17.03 at frame 25, exactly when I wanted the road to come back. Detail was worse on seven of nine come-up frames and the return missed at 0.0614 / 0.578. The ordinary config kept one forward hook at block 0 and the monotone guidance curve.

## Can a kernel pick the spacing? Can a warp organize it?

Before building E-3 I worked out the shortest spacing the proposed difference-of-Gaussians kernel could generate. At block 0 the corrected minimum was 71.1 image pixels. My instrument stops at 64.0. The only late blocks with a measurable window had already gone almost blank when hooked. That's a reason to close the proposed sweep without rendering it. It says something different from running a valid kernel test and measuring zero effect.

E-9r was cheap enough to ask separately. Arm A already makes some repeating texture. Can the arm B map organize *that*? On flat gray, log-polar concentration fell when I added the map. Maybe the map just needed actual edges. I tried the oriented forest scene and its radial fraction fell too. Across the two probes, 96 frames gave me no upward separation on the measures I picked beforehand. Arm B stayed a separate negative result.

Then I put a four-cell hex lattice directly into block 0. The forward signal got the layout, the skip copies stayed clean, and the final image still lost the layout. That made the persistent activation field (E-7) and several other activation-space proposals hard to justify. The experiment tested one injection site. Why later layers lose the layout is still my interpretation. [Eight seeds and the scrambled control](lattice-probe.md)

## What could still change the pixels?

**E-8. Move the starting noise.** The old drift implementation wasn't available, so this one had to be built and checked before any render counted. I used spherical interpolation between noise anchors. At three anchors the calibrated phase step was 0.209062. Six implementation assertions passed and five deliberately broken cases were caught. Then I spent 130 frames checking what it did.

On the holds, mean frame-to-frame change rose from 3.686 to 31.726, three seeds per setting. Correlation stayed high enough for consecutive patterns to line up. But the slope stayed up at only two seeds. When I let a full drift arc come down, it missed closure on both numbers and its peak spacing hit the 3.7 px instrument floor. The main arc kept drift off. The old drift-off reading of 6.15 came from the JPG scene. Re-scoring those old frames reproduced it. The PNG run needs its own baseline.

**C-3. Split the guidance by frequency.** This time I changed the signal that steers denoising. Before rendering I corrected a test that asked a Gaussian filter to retain 99 percent of the low band, even at scales where that was mathematically impossible. The replacement check asked for low-band retention of at least 0.5 and low-over-high selectivity of at least 10. The implementation gave 0.5550 and 22.65. That's a correction to the filter test. The image targets stayed where they were.

The nine-frame probe was supposed to decide whether a 66-frame follow-up was worth doing. High-band energy shifted with the setting, but the gap was 1.58 times the pooled within-setting SD. I needed 3 times. Fine-detail scores stayed between 0.008 and 0.013. That ended C-3 before the longer run.

**C-4. Alternate the gain over time.** This was the remaining conditional idea for motion. It would install an external rhythm, so even a positive result would have to name that assumption. It wasn't requested or run. E-7 stayed retired, and neither the late-block kernel nor arm B was promoted.

## Where I stopped

I kept the one config that could make the whole trip back to frame 00, then checked that return at three seeds. I could claim a temporary rise in repetition. I couldn't claim that I picked its spacing, that it had a center, or that the peak kept evolving on its own. A blank frame, an unmeasurable kernel and a moving peak that ends at a different sample of the road are three different outcomes. Keeping them separate tells me more about the model than calling all three “geometry failed.”

[Deliverable measurements and reproduction commands](results.md)
