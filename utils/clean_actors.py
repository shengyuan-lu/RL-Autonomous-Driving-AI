import carla
from carla import Client

def clean_actors():
    client = Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # destroy only traffic cone and vehicle actors
    obstacle_list = world.get_actors().filter('static.prop.*')
    vehicle_list = world.get_actors().filter('vehicle.*')

    for actor in vehicle_list:
        print(f"Destroying actor: {actor.id}")
        actor.destroy()

    for actor in obstacle_list:
        print(f"Destroying actor: {actor.id}")
        actor.destroy()