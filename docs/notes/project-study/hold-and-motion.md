# How do I tell if a pattern is moving?

Watch frame 12 then 13. Different pictures. But the dose changed between them too, so this doesn't tell me the peak can keep itself going. I need a hold. Pin the controls at the top, keep feeding the loop, THEN compare neighboring frames.

Mean change tells me how different consecutive frames are on a 0 to 255 gray scale. Slope tells me whether that change is dying out across the hold. Neither one knows what a moving object is. For that I slide one frame against the next and look at the best match and where it lands.

- Pattern slides sideways = good match at an offset. If the offset keeps the same direction I've got evidence of travel.
- Pattern changes in place = good match near zero offset. It can still get brighter, weaker or rearrange a little.
- Every frame turns into fresh static = lots of change, poor match. My churn control reads 0.158 on peak correlation.

With ordinary noise the change decayed at all three seeds. With drift the frames changed more and still matched at 0.8922, but only two of three sustained the change. The verdict on those holds was standing. A bigger number on motion didn't give me a traveling wave.

[Hold results](../../experiments/results.md) · [M-3](../../experiments/measurements.md)
