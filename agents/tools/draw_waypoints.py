import carla
from global_route_planner import GlobalRoutePlanner


def draw_waypoints(world, start_loc, end_loc):
    sampling_resolution = 2
    map = world.get_map()
    grp = GlobalRoutePlanner(map, sampling_resolution)

    w1 = grp.trace_route(start_loc, end_loc) # there are other funcations can be used to generate a route in GlobalRoutePlanner.
    #print(w1)
    i = 0
    for w in w1:
        if i % 10 == 0:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
            persistent_lines=True)
        else:
            world.debug.draw_string(w[0].transform.location, 'O', draw_shadow=False,
            color = carla.Color(r=0, g=0, b=255), life_time=1000.0,
            persistent_lines=True)
        i += 1
        
