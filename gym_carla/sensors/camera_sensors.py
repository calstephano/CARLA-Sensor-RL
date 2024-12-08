import carla
import threading
import pygame
import numpy as np
from skimage.transform import resize
from gym_carla.display.display_utils import grayscale_to_display_surface

class CameraSensors:
  def __init__(self, world, obs_size, display_size):
    self.world = world
    self.obs_size = obs_size
    self.display_size = display_size
    self.cameras = []
    self.camera_img = np.zeros((4, obs_size, obs_size), dtype=np.uint8)  # Cache for resized images
    self.vehicle = None
    self.lock = threading.Lock()

    # Import camera blueprint library from CARLA
    self.camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # Set up camera sensor attributes
    self.camera_bp.set_attribute('image_size_x', str(obs_size))   # Set horizontal resolution
    self.camera_bp.set_attribute('image_size_y', str(obs_size))   # Set vertical resolution
    self.camera_bp.set_attribute('fov', '110')            				# Set field of view
    self.camera_bp.set_attribute('sensor_tick', '0.02')       		# Set time (seconds) between sensor captures

    # Define positions for each camera
    self.camera_positions = [
      carla.Transform(carla.Location(x=1.5, z=1.5)),                          											# Front view
      carla.Transform(carla.Location(x=0.7, y=0.9, z=1), carla.Rotation(pitch=-35.0, yaw=134.0)),   # Left-back diagonal view
      carla.Transform(carla.Location(x=-1.5, z=1.5), carla.Rotation(yaw=180.0)),            				# Rear view
      carla.Transform(carla.Location(x=0.7, y=-0.9, z=1), carla.Rotation(pitch=-35.0, yaw=-134.0))  # Right-back diagonal view
    ]

  def spawn_and_attach(self, vehicle):
    self.vehicle = vehicle

    for i, position in enumerate(self.camera_positions):
      # Spawn camera sensor and attach it to the vehicle
      camera = self.world.spawn_actor(self.camera_bp, position, attach_to=self.vehicle)

      # Listen for data
      camera.listen(lambda data, idx=i: self._get_camera_img(data, idx))

      self.cameras.append(camera)

  def _get_camera_img(self, data, index):
    """Process and store camera sensor data as a preprocessed grayscale image."""
    # Convert raw data to numpy array and reshape to a 2D image with RGB channels
    array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
    array = np.reshape(array, (data.height, data.width, 4))[:, :, :3]
    
    # Convert to grayscale
    array = np.mean(array, axis=2).astype(np.uint8)

    # Resize the grayscale image to the observation size
    resized_array = resize(array, (self.obs_size, self.obs_size), preserve_range=True).astype(np.uint8)
    
    # Safely update the corresponding index in the cache
    with self.lock:
      self.camera_img[index] = resized_array

  def render_camera_img(self, display):
    with self.lock:
      for i, camera_position in enumerate(self.camera_positions):
        camera_surface = grayscale_to_display_surface(self.camera_img[i], self.display_size)
        display.blit(camera_surface, (self.display_size * (i + 2), 0))

  def stop(self):
    self.stopped = True
    for camera in self.cameras:
      camera.stop()
    self.cameras = []