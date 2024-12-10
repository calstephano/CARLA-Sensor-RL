from __future__ import division

import sys
import open3d as o3d
import numpy as np
import pygame
import random
import time
import threading

from PIL import Image

import gymnasium as gym
from gymnasium import spaces
import carla

# Local module imports
from gym_carla.env.route_planner import *
from gym_carla.env.env_utils import *
from gym_carla.display import *
from gym_carla.dynamic_actors import *
from gym_carla.sensors import *

class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params, writer=None):
    ...
    self.total_step = 0
    self.display_size = params['display_size']
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/0.125)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self.ego_vehicle_filter = params['ego_vehicle_filter']
    self.writer = writer

    # Action space
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0],
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer

    # Observation space
    self.observation_space = spaces.Dict({
      'state': spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(4,),
        dtype=np.float32),
      'camera': spaces.Box(
        low=0, high=255,
        shape=(4, self.obs_size, self.obs_size),
        dtype=np.uint8),
    })

    # Connect to CARLA server and get world object
    print("\nConnecting to CARLA server...")
    client = carla.Client('localhost', params['port'])
    client.set_timeout(4000.0)
    self.world = client.load_world(params['town'])
    print("CARLA server connected!")

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = get_vehicle_spawn_points(self.world)
    print(f"Retrieved {len(self.vehicle_spawn_points)} vehicle spawn points.")
    self.walker_spawn_points = generate_walker_spawn_points(self.world, self.number_of_walkers)
    print(f"Generated {len(self.walker_spawn_points)} valid walker spawn points out of {self.number_of_walkers} requested.")

    # Initialize sensors
    self.collision_detector = CollisionDetector(self.world)
    self.camera_sensors = CameraSensors(self.world, self.obs_size, self.display_size)

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Initialize Pygame display
    self.pygame_manager = PygameManager(self.world, self.display_size, self.obs_range, self.d_behind, self.display_route)
    self.pygame_manager.camera_sensors = self.camera_sensors

  def reset(self, seed = None, options = None):
    # Disable sync mode
    set_synchronous_mode(self.world, False)

    # Reset environment objects
    self._reset_environment_objects()

    # Spawn surrounding vehicles and walkers
    random.shuffle(self.vehicle_spawn_points)
    vehicles_spawned = spawn_random_vehicles(self.world, self.vehicle_spawn_points, self.number_of_vehicles)
    walkers_spawned = spawn_walkers(self.world, self.walker_spawn_points, self.number_of_walkers)
    print(f"Successfully spawned {walkers_spawned} out of {self.number_of_walkers} walkers.")

    # Get polygon lists of surronding vehicles and walkers
    self.vehicle_polygons = []
    vehicle_poly_dict = get_actor_polygons(self.world, 'vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = get_actor_polygons(self.world, 'walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_vehicle = spawn_ego_vehicle(
      self.world, self.vehicle_spawn_points, self.vehicle_polygons, 
      self.max_ego_spawn_times, self.ego_vehicle_filter, 
    )
    if not ego_vehicle:
      print("Failed to spawn ego vehicle. Resetting environment.")
      return self.reset()
    self.ego = ego_vehicle

    # Spawn and attach sensors
    self.collision_detector.spawn_and_attach(self.ego)
    self.collision_detector.clear_collision_history()
    self.camera_sensors.spawn_and_attach(self.ego)

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    set_synchronous_mode(self.world, True, fixed_delta_seconds=self.dt)

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Set ego information for birdeye render
    self.pygame_manager.set_hero(self.ego, self.ego.id)

    info = self._get_info()

    return self._get_obs(), info

  def step(self, action):
    # Pass updated polygons and waypoints to PygameManager for rendering
    self.pygame_manager.vehicle_polygons = self.vehicle_polygons
    self.pygame_manager.walker_polygons = self.walker_polygons
    self.pygame_manager.waypoints = self.waypoints

    # Pass updated camera
    self.pygame_manager.update_display()

    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    if acc > 0:
      throttle = np.clip(acc/3,0,1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip(-acc/8,0,1)

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    # Tick the world
    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = get_actor_polygons(self.world, 'vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)

    walker_poly_dict = get_actor_polygons(self.world, 'walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # Route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # Get reward and log components
    total_reward, reward_components = self._get_reward(self.total_step)
    if self.writer:
        for key, value in reward_components.items():
            self.writer.add_scalar(f"rewards/{key}", value, self.total_step)

    # Check termination conditions
    terminated = self._terminal()

    # TODO episode truncates if last waypoint is reached
    truncated = False

    # Update timesteps
    self.time_step += 1   # Episode timestep
    self.total_step += 1  # Global timestep

    # Prepare info dictionary
    info = self._get_info()
    info["reward_components"] = reward_components  # Include reward components for debugging

    return (self._get_obs(), total_reward, self._terminal(), truncated, info)

  def _get_obs(self):
    """Get the observations."""

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi                          # Convert yaw to radians
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y) # Calculate lateral distance and heading error to lane center
    delta_yaw = np.arcsin(np.cross(w,
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front], dtype=np.float32)

    # Retrieve optimized camera images for RL
    camera_images = self.camera_sensors.camera_img

    obs = {
      'state':state,
      'camera':camera_images,
    }

    return obs

  def _get_reward(self, step):
    """Calculate the reward."""
    # Extract state variables
    obs = self._get_obs()
    lateral_dis, delta_yaw, speed, vehicle_front = obs['state']

    max_delta_yaw = np.pi / 4

    # Dynamically retrieve lane width from the map
    ego_location = self.ego.get_location()
    ego_waypoint = self.world.get_map().get_waypoint(ego_location)
    lane_width = ego_waypoint.lane_width if ego_waypoint else 2.0

    # Reward components
    r_lane = -abs(lateral_dis / lane_width)    # Penalize deviation from lane center
    r_heading = -abs(delta_yaw / (np.pi / 4))  # Penalize large heading errors

    # Speed reward
    r_speed = -abs((speed - self.desired_speed) / self.desired_speed)  # Penalize speed deviations
    if abs(lateral_dis) > lane_width * 0.5:                            # Dynamic speed reward
        r_speed -= 1                                                   # Penalize high speed when off-center

    r_collision = -1 if self.collision_detector.get_latest_collision_intensity() else 0  # Heavy collision penalty

    # Penalize abrupt yaw changes
    r_smooth_yaw = -abs(delta_yaw - getattr(self, 'previous_yaw', delta_yaw)) / max_delta_yaw
    self.previous_yaw = delta_yaw

    # Penalize lateral acceleration
    current_steer = self.ego.get_control().steer
    v = self.ego.get_velocity()
    lspeed_lon = np.dot([v.x, v.y], [np.cos(delta_yaw), np.sin(delta_yaw)])
    r_lateral_acc = -abs(current_steer) * (lspeed_lon**2 / self.desired_speed**2)

    # Reward for waypoint progress
    progress_reward = 0
    if self.waypoints:
        ego_x, ego_y = get_pos(self.ego)
        waypoint_x, waypoint_y = self.waypoints[0][:2]

        # Reward progress only if the vehicle is within a reasonable distance from the lane center
        if abs(lateral_dis) <= lane_width * 0.5:  # Ensure vehicle is within half the lane width
            # Reward for reducing distance to the next waypoint
            distance_to_waypoint = np.linalg.norm([ego_x - waypoint_x, ego_y - waypoint_y])
            previous_distance = getattr(self, 'previous_distance_to_waypoint', float('inf'))
            progress_reward = 5 if distance_to_waypoint < previous_distance else -1
            self.previous_distance_to_waypoint = distance_to_waypoint

            # Bonus for reaching the waypoint
            if distance_to_waypoint < 1.0:  # Within 1 meter
                progress_reward += 10
                self.waypoints.pop(0)
        else:
            # Penalize for making progress while off-lane
            progress_reward = -5

    # Combine rewards
    total_reward = (
        10 * r_lane +
        5 * r_heading +
        2 * r_speed +
        50 * r_collision +
        2 * r_smooth_yaw +
        # 2 * r_smooth_steering +
        0.5 * r_lateral_acc +
        progress_reward
    )
    total_reward = np.clip(total_reward, -100, 100)

    # Log rewards
    if self.writer:
        reward_components = {
            "lane_reward": r_lane,
            "heading_reward": r_heading,
            "speed_reward": r_speed,
            "collision_reward": r_collision,
            "smooth_yaw_penalty": r_smooth_yaw,
            "lateral_acceleration_penalty": r_lateral_acc,
            "progress_reward": progress_reward,
            "total_reward": total_reward
        }
        for key, value in reward_components.items():
            self.writer.add_scalar(f"rewards/{key}", value, self.total_step)

    return total_reward, reward_components

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if self.collision_detector.get_latest_collision_intensity():
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If out of lane
    dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    if abs(dis) > self.out_lane_thres:
      return True

    return False

  def _get_info(self):
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }
    return info
  
  def _reset_environment_objects(self):
    # Delete actors
    clear_all_actors(self.world, [
      'sensor.other.collision', 'sensor.camera.rgb',
      'vehicle.*', 'controller.ai.walker', 'walker.*'
    ])

    # Clear sensor objects
    self.collision_detector.collision_detector = None
    self.camera_sensors.camera_sensors = None

  def close(self):
    # Stop listening for data
    self.collision_detector.stop()
    self.camera_sensors.stop()

    # Remove objects
    self._reset_environment_objects()
