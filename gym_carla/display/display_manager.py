import pygame
from render import *
from gym_carla.display import *
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import Video
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

  def update_display(self, camera_sensors):
    """Render and update the display with bird's-eye view and camera view."""
    # Render bird's-eye view
    birdeye = self._get_birdeye()
    birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
    self.display.blit(birdeye_surface, (0, 0))

    # Render camera view
    camera_sensors.render_camera_img(self.display)

    # Display both
    pygame.display.flip()


class VideoWrapper(gym.Env):
  """
  Wrapper for recording video during simulation.
  Captures frames from the environment and saves them as a video.
  """
  def __init__(self, env, video_dir="videos", fps=30):
    """
    Args:
        env (gym.Env): The environment to wrap.
        video_dir (str): Directory where videos will be saved.
        fps (int): Frames per second for the video recording.
    """
    self.env = env
    self.video_dir = video_dir
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
    video = Video(frames=np.array(self.frames), fps=self.fps)
    os.makedirs(self.video_dir, exist_ok=True)
    video.save(f"{self.video_dir}/episode_{np.random.randint(10000)}.mp4")
    self.env.close()

