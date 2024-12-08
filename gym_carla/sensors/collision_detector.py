import carla
import numpy as np

class CollisionDetector:
  def __init__(self, world):
    self.world = world
    self.vehicle = None
    self.collision_history = []

    # Import collision blueprint library from CARLA
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

  def spawn_and_attach(self, vehicle):
    self.vehicle = vehicle

    # Spawn the collision detector and attach it to the vehicle
    self.collision_detector = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.vehicle)

    # Listen for events
    self.collision_detector.listen(lambda event: self._on_collision(event))

  def _on_collision(self, event):
    impulse = event.normal_impulse
    intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    self.collision_history.append(intensity)

    # Keep collision history length to the last recorded collision event
    max_history_length = 1
    if len(self.collision_history) > max_history_length:
      self.collision_history.pop(0)

    # Log and print details of the collision
    other_actor = event.other_actor
    actor_type = other_actor.type_id if other_actor else "Unknown"
    impulse_magnitude = np.linalg.norm([impulse.x, impulse.y, impulse.z])
    print(f"[TERMINATE] Collision detected!")
    print(f"  - Other actor type: {actor_type}")
    print(f"  - Collision intensity: {impulse_magnitude}")
    print(f"  - Collision location: {event.transform.location}")

  def get_data(self):
    # Return latest collision data
    return self.collision_data

  def get_latest_collision_intensity(self):
    # Return latest collision intensity if available
    return self.collision_history[-1] if self.collision_history else None

  def clear_collision_history(self):
    # Clear collision history
    self.collision_history = []
    
  def stop(self):
    self.collision_detector.stop()