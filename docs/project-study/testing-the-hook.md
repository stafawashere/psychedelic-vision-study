# How do I know I poked the right thing?

Say my hook test makes its own fake noise anchors and checks that they're different. The test might pass every time. Then I accidentally make the engine generate the same anchor three times. My test still passes because it never looked at what the engine drew. That's the kind of bug the drift check ran into.

I want a test that goes through the same entry point the real frame uses. Change the production anchors, leave the test alone and see which check catches it. A crash doesn't tell me much if I was trying to test continuity. I want the render path to stay valid long enough to give me the WRONG behavior, then see the assertion fail for that reason.

The split-guidance setting has another version of this problem. A value can come from a YAML file or the default I get by constructing the config in Python. Checking only the YAML path leaves a second way for an experiment to turn itself on. I check both. The main config should only split guidance if I deliberately ask it to.

> first break the exact behavior the check claims to catch. then restore it and check again.

[Drift checks](../../tests/test_noise_drift.py) · [Guidance checks](../../tests/test_guidance_split.py)
