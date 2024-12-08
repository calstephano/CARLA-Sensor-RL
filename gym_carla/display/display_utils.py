import math
import numpy as np
import pygame
import skimage
from gym_carla.display import *


def display_to_rgb(display, obs_size):
  """
  Transform image grabbed from pygame display to an rgb image uint8 matrix

  :param display: pygame display input
  :param obs_size: rgb image size
  :return: rgb image uint8 matrix
  """
  rgb = np.fliplr(np.rot90(display, 3))                      # Flip to regular view
  rgb = skimage.transform.resize(rgb, (obs_size, obs_size))  # Resize
  rgb = rgb * 255
  return rgb


def rgb_to_display_surface(rgb, display_size):
  """
  Generate pygame surface given an RGB image uint8 matrix.

  :param rgb: RGB image uint8 matrix.
  :param display_size: Display size.
  :return: Pygame surface.
  """
  surface = pygame.Surface((display_size, display_size)).convert()
  display = skimage.transform.resize(rgb, (display_size, display_size), preserve_range=True).astype(np.uint8)
  display = np.flip(display, axis=1)
  display = np.rot90(display, 1)
  pygame.surfarray.blit_array(surface, display)
  return surface


def grayscale_to_display_surface(gray, display_size):
  """
  Convert a grayscale image into a Pygame-compatible surface for rendering.

  Note:
  - Grayscale is converted to RGB (3 channels) for visualization purposes only (Pygame requirement)
  - This does not impact RL tasks, where grayscale is retained internally for efficiency.

  :param gray: Grayscale image as a NumPy array (uint8 matrix).
  :param display_size: Display size.
  :return: Pygame surface.
  """
  # Convert grayscale to RGB (3 channels)
  rgb = np.stack((gray, gray, gray), axis=2)

  # Render through rgb_to_display_surface function
  return rgb_to_display_surface(rgb, display_size)


