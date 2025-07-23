# Tello Obstacle Detection

This project focuses on integrating OpenCV with Monocular RGB depth estimation to avoid obstacles indoor environment.

## Table of Content

## Design Rationale

### Navigation Strategy
Why a pre‑defined grid?

Sensor constraints
The Tello EDU has only a monocular camera and no depth, LiDAR or IMU strong enough for reliable SLAM. Without adequate onboard sensors, any on‑the‑fly map‑building would drift or fail. Additionally, the drone cannot know its real‑world coordinates at take‑off or during flight as it does not have a GPS. A fixed grid map gives us predefined waypoints in a common reference frame, so the drone “knows” where to start, where to go, and when it’s arrived.

Deterministic waypoints
By splitting our rectangular arena into a fixed grid with drawing an “X” inside each cell, we know exactly where every corner and cell‑center is in advance. This guarantees reproducible flight paths.

Trade‑offs

Pros: Low computation, zero mapping drift, easy debugging.

Cons: Not effective to real-world uncontrolled environment.

## Reference

[NetworkX](https://networkx.org/) under the BSD‑3‑Clause License [NetworkX.LICENSE.txt](NetworkX.LICENSE.txt)
