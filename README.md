# AIMS SLAM - CMU AI MakerSpace

This repository provides a ROS2-based SLAM development framework for robotics projects at the CMU AI MakerSpace.  

It integrates:
- RealSense D435i camera interface for ROS2.
- ORB-SLAM3 for real-time RGB-D SLAM.

This setup is intended to enable quick deployment and experimentation with SLAM systems in the AI MakerSpace projects.

---

## What is ORB-SLAM?

ORB-SLAM3 is a feature-based SLAM (Simultaneous Localization and Mapping) framework that leverages ORB (Oriented FAST and Rotated BRIEF) features for:

- Robust feature extraction.
- Real-time camera motion tracking.
- Keyframe-based sparse mapping.
- Loop closure for correcting long-term drift.

In the context of RGB-D SLAM, the system utilizes:
- RGB images for feature extraction.
- Depth images for accurate map construction.
- IMU data for motion prediction and robustness.

For further details on the ORB-SLAM3 ROS2 integration, refer to:  
➡️ https://github.com/zang09/ORB_SLAM3_ROS2

---

## Package Overview

This repository primarily contains:

### `aims_slam`

A ROS2 package for:
- Capturing RGB, Depth, and IMU data from Intel RealSense D435i.
- Publishing ROS2 topics compatible with ORB-SLAM3.

---

## Tested Platforms

This repository has been tested and validated on the following configurations:

| ROS2 Distribution | Ubuntu Version  |
|------------------|-----------------|
| Foxy              | 20.04           |
| Humble            | 20.04           |

---

## Setup Instructions

### 1. Clone the Repository

```bash
mkdir -p ~/aims_slam_ws/src
cd ~/aims_slam_ws/src
git clone https://github.com/CMU-AI-Maker-Space/aims_slam.git

