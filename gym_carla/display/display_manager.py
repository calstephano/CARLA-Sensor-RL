import pygame
from .display_utils import *
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import Video
from gym_carla.sensors import CameraSensors
import os

class PygameManager:
  def __init__(self, world, display_size, obs_range, d_behind, display_route):
    self.display_size = display_size
    self.obs_range = obs_range
    self.d_behind = d_behind
    self.display_route = display_route
    self.world = world
    self.vehicle_polygons = None
    self.walker_polygons = None
    self.waypoints = None
    self.camera_sensors = None
    
    pygame.init()
    self.display = pygame.display.set_mode(
      (self.display_size * 6, self.display_size),
      pygame.HWSURFACE |  pygame.DOUBLEBUF
    )

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
        'screen_size': [self.display_size, self.display_size],
        'pixels_per_meter': pixels_per_meter,
        'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def set_hero(self, hero_actor, hero_id):
    """Set the hero (ego vehicle) in the BirdeyeRender."""
    self.birdeye_render.set_hero(hero_actor, hero_id)

  def _get_birdeye(self):
    """Retrieve and format the bird's-eye view data into RGB"""
    self.birdeye_render.vehicle_polygons = self.vehicle_polygons
    self.birdeye_render.walker_polygons = self.walker_polygons
    self.birdeye_render.waypoints = self.waypoints

    birdeye_render_types = ['roadmap', 'actors']
    if self.display_route:
      birdeye_render_types.append('waypoints')
    
    self.birdeye_render.render(self.display, birdeye_render_types)
    birdeye = pygame.surfarray.array3d(self.display)
    birdeye = birdeye[0:self.display_size, :, :]
    birdeye = display_to_rgb(birdeye, self.display_size)
    
    return birdeye

  def update_display(self):
    """Render and update the display with bird's-eye view and camera view."""
    # Render bird's-eye view
    birdeye = self._get_birdeye()
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    # Render camera view
    self.camera_sensors.render_camera_img(self.display)

    # Display both
    pygame.display.flip()


class VideoWrapper(gym.Env):
  """
  Wrapper for recording video during simulation.
  Captures frames from the environment and saves them as a video.
  """
  def __init__(self, env, playback_dir, fps=30):
    """
    Args:
        env (gym.Env): The environment to wrap.
        playback_dir (str): Directory where videos will be saved.
        fps (int): Frames per second for the video recording.
    """
    self.env = env
    self.playback_dir = playback_dir
    self.fps = fps
    self.frames = []

  def reset(self):
    """
    Reset the environment and initialize frame storage.
    """
    obs = self.env.reset()
    self.frames = []
    return obs

  def step(self, action):
    """
    Take an action in the environment and capture the frame.
    """
    obs, reward, terminated, truncated, info = self.env.step(action)

    frame = self.env.render(mode="rgb_array")
    self.frames.append(frame)

    return obs, reward, terminated, truncated, info

  def close(self):
    """
    Save the frames as a video when the episode finishes.
    """
    # Increment the episode counter
    self.episode_counter += 1
    
    # Create a new directory for each training session if needed
    video_folder = os.path.join(self.playback_dir, f"training_session_{self.episode_counter}")
    os.makedirs(video_folder, exist_ok=True)

    # Save the frames as a video, naming each episode sequentially
    video = Video(frames=np.array(self.frames), fps=self.fps)
    video.save(f"{video_folder}/episode_{self.episode_counter:04d}.mp4")

    # Clean up the environment
    self.env.close()
