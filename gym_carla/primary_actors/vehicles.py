import random
import numpy as np
import carla
import time


def get_vehicle_spawn_points(world):
  """Retrieve a list of spawn points for vehicles in the CARLA world.

  Args:
    world (carla.World): The CARLA simulation world.

  Returns:
    list: List of carla.Transform objects representing vehicle spawn points.
  """
  return list(world.get_map().get_spawn_points())


def try_spawn_ego_vehicle_at(world, ego_bp, vehicle_polygons, transform):
  """
  Try to spawn the ego vehicle at a specific transform.

  Args:
    world (carla.World): The CARLA simulation world.
    ego_bp (carla.ActorBlueprint): The blueprint of the ego vehicle.
    vehicle_polygons (list): List of surrounding vehicle polygons to check for overlap.
    transform (carla.Transform): The transform (location and rotation) to spawn the vehicle.

  Returns:
    carla.Vehicle or None: The spawned ego vehicle, or None if spawning fails.
  """
  # Check if ego position overlaps with surrounding vehicles
  for poly in vehicle_polygons[-1].values():
    poly_center = np.mean(poly, axis=0)
    ego_center = np.array([transform.location.x, transform.location.y])
    if np.linalg.norm(poly_center - ego_center) < 8:
        return None  # Overlap detected, spawn failed

  # Try to spawn the ego vehicle
  vehicle = world.try_spawn_actor(ego_bp, transform)
  return vehicle  # Return the vehicle if successfully spawned, or None



def spawn_ego_vehicle(world, vehicle_spawn_points, ego_bp, vehicle_polygons, max_ego_spawn_times):
  """
  Spawn the ego vehicle at a random location.
  
  Args:
    world (carla.World): The CARLA simulation world.
    vehicle_spawn_points (list): List of spawn points for vehicles.
    ego_bp (carla.ActorBlueprint): The blueprint of the ego vehicle.
    vehicle_polygons (list): List of surrounding vehicle polygons to check for overlap.
    max_ego_spawn_times (int): Maximum number of attempts to spawn the ego vehicle.
  
  Returns:
    carla.Vehicle or None: The spawned ego vehicle, or None if spawning fails.
  """
  ego_spawn_times = 0
  while ego_spawn_times <= max_ego_spawn_times:
    transform = random.choice(vehicle_spawn_points)
    vehicle = try_spawn_ego_vehicle_at(world, ego_bp, vehicle_polygons, transform)
    if vehicle:
      print(f"Ego vehicle spawned successfully at {transform.location}.")
      return vehicle  # Return the spawned ego vehicle instance
    ego_spawn_times += 1
    time.sleep(0.1)

  print("Failed to spawn ego vehicle after maximum attempts.")
  return None


def try_spawn_random_vehicle_at(world, transform, number_of_wheels=[4]):
  """
  Try to spawn a surrounding vehicle at a specific transform with a random blueprint.

  Args:
    world (carla.World): The CARLA simulation world.
    transform (carla.Transform): The CARLA transform object for spawn location.
    number_of_wheels (list): List specifying the number of wheels for the vehicle.

  Returns:
    bool: True if the vehicle was successfully spawned, False otherwise.
  """
  print(f"Attempting to spawn vehicle at: {transform.location}")
  blueprint = create_vehicle_blueprint(world, 'vehicle.*', number_of_wheels=number_of_wheels)

  blueprint.set_attribute('role_name', 'autopilot')
  vehicle = world.try_spawn_actor(blueprint, transform)
  if vehicle is not None:
    vehicle.set_autopilot(enabled=True, tm_port=4050)
    return True

  return False


def spawn_random_vehicles(world, spawn_points, number_of_vehicles, number_of_wheels=[4]):
  """
  Spawn a specified number of surrounding vehicles at given spawn points.

  Args:
    world (carla.World): The CARLA simulation world.
    spawn_points (list): List of carla.Transform objects for spawn locations.
    number_of_vehicles (int): Total number of vehicles to spawn.
    number_of_wheels (list, optional): List specifying the number of wheels for the vehicles.

  Returns:
    int: The number of vehicles successfully spawned.
  """
  random.shuffle(spawn_points)
  count = number_of_vehicles
  if count > 0:
    for spawn_point in spawn_points:
      if try_spawn_random_vehicle_at(world, spawn_point, number_of_wheels=[4]):
        print(f"Successfully spawned vehicle at {spawn_point.location}.")
        count -= 1
      else:
        print(f"Failed to spawn vehicle at {spawn_point.location}.")
      if count <= 0:
        break
  while count > 0:
    if try_spawn_random_vehicle_at(world, random.choice(spawn_points), number_of_wheels=[4]):
      count -= 1


def create_vehicle_blueprint(world, actor_filter, color=None, number_of_wheels=[4]):
  """
  Create the blueprint for a specific actor type.

  Args:
    world (carla.World): The CARLA simulation world.
    actor_filter (str): A string indicating the actor type, e.g., 'vehicle.lincoln*'.
    color (str, optional): Specific color for the vehicle (default is random if not provided).
    number_of_wheels (list, optional): List of acceptable numbers of wheels for the vehicle (default is [4]).

  Returns:
    carla.ActorBlueprint: The blueprint object for the specified actor type.
  """
  blueprints = world.get_blueprint_library().filter(actor_filter)
  blueprint_library = []
  for nw in number_of_wheels:
    blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
  bp = random.choice(blueprint_library)
  if bp.has_attribute('color'):
    if not color:
      color = random.choice(bp.get_attribute('color').recommended_values)
    bp.set_attribute('color', color)
  return bp