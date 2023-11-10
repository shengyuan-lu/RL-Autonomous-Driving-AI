import carla
import math

def is_within_distance(location, actor, radius=10.0, ):
    """
    params:
        - location: trigger box location
        - actor: actor to be calculated

    Return true if the actor is within proper distance to location
    """

    distance = math.sqrt(
                        (location.x - actor.get_location().x) ** 2 +
                        (location.y - actor.get_location().y) ** 2 +
                        (location.z - actor.get_location().z) ** 2
    )

    return distance < radius