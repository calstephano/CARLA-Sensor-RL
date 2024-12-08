import numpy as np


def get_actor_polygons(world, filt):
  """
  Get the bounding box polygon of actors.

  Args:
    world (carla.World): The CARLA simulation world.
    filt (str): The filter indicating what type of actors to look at.

  Returns:
    dict: A dictionary containing the bounding boxes of specific actors.
  """
  actor_poly_dict = {}
  flagged_actors = set()  # Keep track of actors with invalid polygons

  for actor in world.get_actors().filter(filt):
    try:
      # Get x, y, and yaw of the actor
      trans = actor.get_transform()
      x = trans.location.x
      y = trans.location.y
      yaw = trans.rotation.yaw / 180 * np.pi

      # Get length and width
      bb = actor.bounding_box
      l = bb.extent.x
      w = bb.extent.y

      # Get bounding box polygon in the actor's local coordinate
      poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()

      # Get rotation matrix to transform to global coordinate
      R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
      
      # Get global bounding box polygon
      poly = np.matmul(R, poly_local).transpose() + np.array([[x, y]] * 4)

      # Check for NaN values and add valid polygons to the dictionary
      if not np.isnan(poly).any():
        actor_poly_dict[actor.id] = poly
      else:
        if actor.id not in flagged_actors:
          print(f"[DEBUG] Invalid polygon detected for actor {actor.id} ({actor.type_id}). Contains NaN values.")
          flagged_actors.add(actor.id)
    except Exception as e:
      print(f"Error processing actor {actor.id}: {e}")
  
  return actor_poly_dict


def clear_all_actors(world, actor_filters):
  """
  Clear all actors matching specific filters from the CARLA world.

  Args:
    world (carla.World): The CARLA world object.
    actor_filters (list of str): List of filters for actor types to remove.
      Examples: ['vehicle.*', 'walker.*', 'sensor.*']
  """
  for actor_filter in actor_filters:
    for actor in world.get_actors().filter(actor_filter):
      if actor.is_alive:
        if actor.type_id == 'controller.ai.walker':
          # Stop AI controllers before destruction
          actor.stop()
        try:
          actor.destroy()
        except Exception as e:
          print(f"Error destroying actor {actor.type_id}, ID: {actor.id}: {e}")
