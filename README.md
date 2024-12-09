# CARLA Reinforcment Learning (RL) with Vision

This repository features a custom CARLA Gymnasium environment, built upon the foundation of this custom [gym environment](https://github.com/cjy1992/gym-carla.git). It has been upgraded to leverage Gymnasium for improved compatibility and functionality, and incorporates Stable-Baselines3 (SB3) for advanced reinforcement learning capabilities, replacing the older frameworks.

## Setup

## How to Train

## Running TensorBoard
TensorBoard is a visualization tool used to monitor training metrics, such as rewards and losses. To launch TensorBoard, open a new terminal and run the following commands from the main directory:

```
conda activate carla
tensorboard --logdir ./tensorboard --port 6006
```

Open your browser and go to:
```
http://localhost:6006
```
