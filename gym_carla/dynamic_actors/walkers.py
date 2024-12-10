import random
import numpy as np
import carla


def generate_walker_spawn_points(world, number_of_walkers):
  """
  Generate spawn points for walkers based on navigation locations in the CARLA world.

  Args:
    world (carla.World): The CARLA simulation world.
    number_of_walkers (int): The number of walker spawn points to generate.

  Returns:
    list: A list of carla.Transform objects representing valid walker spawn points.
  """
  walker_spawn_points = []
  for i in range(number_of_walkers):
    spawn_point = carla.Transform()
    loc = world.get_random_location_from_navigation()
    if loc is not None:
      if not (np.isnan(loc.x) or np.isnan(loc.y) or np.isnan(loc.z)):
        print(f"Valid location: {loc}")
        spawn_point.location = loc
        walker_spawn_points.append(spawn_point)
      else:
        print("Location contains NaN values!")
    else:
      print("No location returned (None).")
      
  return walker_spawn_points

def try_spawn_random_walker_at(world, transform):
  """Try to spawn a walker at specific transform with random blueprint.

  Args:
    world: the CARLA simulation world.
    transform: the carla transform object.

  Returns:
    Bool indicating whether the spawn is successful.
  """
  walker_bp = random.choice(world.get_blueprint_library().filter('walker.*'))
  # Set as not invincible
  if walker_bp.has_attribute('is_invincible'):
    walker_bp.set_attribute('is_invincible', 'false')
  walker_actor = world.try_spawn_actor(walker_bp, transform)

  if walker_actor is not None:
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    walker_controller_actor = world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
    
    # Start walker
    walker_controller_actor.start()

    # Set walk to random point
    walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
    
    # Random max speed
    walker_controller_actor.set_max_speed(1 + random.random())  # Max speed between 1 and 2 (default is 1.4 m/s)
    
    return True
  return False


def spawn_walkers(world, spawn_points, number_of_walkers, max_retries=10):
  """
  Spawn a specified number of walkers at given spawn points.

  Args:
    world (carla.World): The CARLA simulation world.
    spawn_points (list): List of carla.Transform objects for spawn locations.
    number_of_walkers (int): Total number of walkers to spawn.
    max_retries (int, optional): Maximum number of retries for spawning failed walkers.

  Returns:
    int: The number of walkers successfully spawned.
  """
  random.shuffle(spawn_points)
  count = number_of_walkers
  retries = 0

  for spawn_point in spawn_points:
    if try_spawn_random_walker_at(world, spawn_point):
      count -= 1
    if count <= 0:
      break

  while count > 0 and retries < max_retries:
    if try_spawn_random_walker_at(world, random.choice(spawn_points)):
      count -= 1
    retries += 1

  return number_of_walkers - count
