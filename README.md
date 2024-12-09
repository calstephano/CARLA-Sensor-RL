# Reinforcment Learning (RL) in CARLA

This repository features a custom CARLA Gymnasium environment, built upon the foundation of this custom [gym environment](https://github.com/cjy1992/gym-carla.git). It has been upgraded to leverage Gymnasium for improved compatibility and functionality, and incorporates Stable-Baselines3 (SB3) for advanced reinforcement learning capabilities, replacing the older frameworks.

## Setup
Clone the repo, download dependencies, and then run
```
git clone https://github.com/calstephano/CARLA-RL.git
cd CARLA-Sensor-RL
git checkout vision-RL
pip3 install -r requirements.txt
pip3 install -e .
export PYTHONPATH=$PYTHONPATH:<path to CARLA>/PythonAPI/carla/dist/carla-<version here>-py3
```

## How to Run
In one terminal, navigate to CARLA's installation directory and host CARLA:

`./CarlaUE4.sh -carla-rpc-port=2000 -norelativemousemode`

In another terminal, run the project:

`python3 run.py`

## Viewing Data
TensorBoard is a visualization tool used to monitor training metrics, such as rewards and losses. To launch TensorBoard, open a new terminal and run the following command in the main directory:

`tensorboard --logdir ./logs --port 6006`

Open your browser and go to:

`http://localhost:6006`
