# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import gymnasium as gym
import gym_carla
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO, SAC

def main():
  # Define environment parameters
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 5,
    'display_size': 256,                        # Screen size
    'max_past_step': 1,                         # The number of past steps to draw
    'dt': 0.1,                                  # Time interval between two frames
    'discrete': True,                           # Whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],           # Discrete value of accelerations
    'discrete_steer': [-0.05, 0.0, 0.05],       # Discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],      # Continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],      # Continuous steering angle range
    'ego_vehicle_filter': 'vehicle.lincoln*',   # Filter for defining ego vehicle
    'port': 2000,                               # Connection port
    'town': 'Town03',                           # Which town to simulate
    'max_time_episode': 1000,                   # Maximum timesteps per episode
    'max_waypt': 12,                            # Maximum number of waypoints
    'obs_range': 32,                            # Observation range (meter)
    'lidar_bin': 0.125,                         # Bin size of LIDAR sensor (meter)
    'd_behind': 12,                             # Distance behind the ego vehicle (meter)
    'out_lane_thres': 2.0,                      # Threshold for out of lane
    'desired_speed': 8,                         # Desired speed (m/s)
    'max_ego_spawn_times': 200,                 # Maximum times to spawn ego vehicle
    'display_route': True,                      # Whether to render the desired route
  }

  # Ask the user to select the training algorithm
  print("\nSelect the training algorithm:")
  print("1. Deep Q-Network (DQN)")
  print("2. Soft Actor-Critic (SAC)")
  print("3. Proximal Policy Optimization (PPO)")
  model_input = input("Enter the model number (1, 2, or 3): ").strip()

  if model_input == "1":
    model_type = 'DQN'
    params['discrete'] = True
  elif model_input == "2":
    model_type = 'SAC'
    params['discrete'] = False
  elif model_input == "3":
    model_type = 'PPO'
    params['discrete'] = False
  else:
    print("Invalid input. Defaulting to DQN.")
    model_type = 'DQN'

  # Initialize base log directory
  base_log_dir = f"./tensorboard/{model_type}"
  os.makedirs(base_log_dir, exist_ok=True)

  # Find the next trial number
  trial_number = len([d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith(model_type)]) + 1

  # Initialize reward directory
  reward_log_dir = f"{base_log_dir}/Reward_Logs_{trial_number}"
  os.makedirs(reward_log_dir, exist_ok=True)

  # Initialize the TensorBoard writer for rewards
  writer = SummaryWriter(log_dir=reward_log_dir)

  # Initialize the environment
  env = gym.make('carla-v0', params=params, writer=writer)

  # Initialize the model
  model = select_model(env, model_type, verbose=1, tensorboard_log=base_log_dir)

  # Train the model
  model.learn(total_timesteps=10000)

  # Test the model
  test_model(model, env)

  # Close the TensorBoard writer
  writer.close()

# Model specifications
def select_model(env, model_type, **kwargs):
  """Select the appropriate model based on the user's choice."""
  if model_type == 'DQN':
    return DQN(
      "MultiInputPolicy",
      env,
      buffer_size=50_000,
      **kwargs
    )
  elif model_type == 'SAC':
    return SAC("MultiInputPolicy", env, buffer_size=50_000, **kwargs)
  elif model_type == 'PPO':
    return PPO("MultiInputPolicy", env, **kwargs)
  else:
    raise ValueError(f"Unsupported model type: {model_type}")

def test_model(model, env, steps=100):
  """Test the trained model."""
  obs, info = env.reset()
  for step in range(steps):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

    # Visualization step
    env.visualize()
    
    if terminated or truncated:
      print("Episode finished.")
      break

if __name__ == '__main__':
  main()
