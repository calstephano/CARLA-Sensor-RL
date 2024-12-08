# This file is modified from the CARLA Python examples library
# Copyright (c) 2020: Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB)

import carla
import numpy as np
import open3d as o3d
from matplotlib import cm

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class LIDARSensor:
  def __init__(self, world):
    self.world = world
    self.vehicle = None
    self.lidar_data = None

    # Set up parameters for LIDAR position
    self.lidar_height = 1.8
    self.lidar_trans = carla.Transform(carla.Location(x=-0.5, z=self.lidar_height))

    # Import LIDAR blueprint library from CARLA
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')

    # Set up LIDAR attributes
    self.lidar_bp.set_attribute('channels', '64.0')
    self.lidar_bp.set_attribute('range', '100.0')
    self.lidar_bp.set_attribute('upper_fov', '15')
    self.lidar_bp.set_attribute('lower_fov', '-25')
    self.lidar_bp.set_attribute('rotation_frequency', str(1.0 / 0.05))
    self.lidar_bp.set_attribute('points_per_second', '500000')

    # Initialize the point list for storing 3D points
    self.point_list = o3d.geometry.PointCloud()

  def spawn_and_attach(self, vehicle):
    self.vehicle = vehicle

    # Spawn LIDAR sensor and attach it to the vehicle
    self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.vehicle)

    # Listen for data
    self.lidar_sensor.listen(lambda data: self._on_data(data))

  def _on_data(self, point_cloud):
    # Get the LIDAR data and convert it to a numpy array.
    p_cloud_size = len(point_cloud)
    lidar_data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    lidar_data = np.reshape(lidar_data, (p_cloud_size, 4))
    
    # Extract the intensity for color mapping
    intensity = lidar_data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
      np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
      np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
      np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Extract 3D points and invert the x-axis for display
    points = lidar_data[:, :-1]
    points[:, :1] = -points[:, :1]

    # Update Open3D point cloud with processed data
    self.point_list.points = o3d.utility.Vector3dVector(points)
    self.point_list.colors = o3d.utility.Vector3dVector(int_color)
    
    # Store processed LIDAR data for further use
    self.lidar_data = lidar_data

  def get_data(self):
    return self.lidar_data