# Tello Obstacle Detection

This project focuses on integrating OpenCV with Monocular RGB depth estimation to avoid obstacles indoor environment.

## Table of Content

## Design Rationale

### Navigation Strategy
#### Problem:
##### Sensor constraints
The Tello EDU has only a monocular camera and no depth, LiDAR or IMU strong enough for reliable SLAM. Without adequate onboard sensors, any on‑the‑fly map‑building would drift or fail. Additionally, the drone cannot know its real‑world coordinates at take‑off or during flight as it does not have a GPS. A fixed grid map gives us predefined waypoints in a common reference frame, so the drone “knows” where to start, where to go, and when it’s arrived.

#### Solution:
##### Deterministic waypoints
By splitting our rectangular arena into a fixed grid with drawing an “X” inside each cell, we know exactly where every corner and cell‑center is in advance. This guarantees reproducible flight paths.

##### Logic
1. Generate pre-defined arena in a rectangle.
2. Allocate nodes with provided distance each other.
3. Connect nodes in diagonal and straight lines.
4. Set the destination node.
5. Calculate shortest path to reach to the destination.
6. Proceed to each node on the path until reaching to the destination.

#### Trade‑offs

Pros: Low computation, zero mapping drift, easy debugging.

Cons: Not effective to real-world uncontrolled environment.

### Obstacle Detection Strategy
#### Problem:
##### Sensor Constraints
As it is mentioned above, the Tello Edu is not equiped enough to detect depth of an object to avoid it. This is because it cannot calculate the size of the object with traditional way of calculation based on angle of view or laser reflection.

#### Solution:
##### MiDaS Monocular Depth Estimation via OpenCV DNN
Using this pre-trained depth estimator model running on the connected laptop via OpenCV’s DNN module gives us a full‑frame, per‑pixel depth map from a single RGB image stream.

#### Trade-offs
Pros: No per‑object setup or real‑world size needed and be able to handle arbitrary scenes.

Cons: Heavy model with less precise for absolute distance.

## Reference

[NetworkX](https://networkx.org/) under the BSD‑3‑Clause License [NetworkX.LICENSE.txt](NetworkX.LICENSE.txt)
