import carla
import random
from shapely.geometry import Polygon

from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)
from agents.navigation.basic_agent import BasicAgent

class DetourAgent(BasicAgent):
    """
    TODO: add descriptions
    """

    def __init__(self, vehicle, waypoints=[], target_speed=20, debug=False):
        self._agent = None
        self._route_assigned = False
        self._vehicle = vehicle
        self._end_location = None
        self._map = self._vehicle.get_world().get_map()
        self._waypoints = waypoints
        self._target_speed = target_speed

        super().__init__(vehicle=vehicle, target_speed=target_speed)


    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        self._end_location = end_location

        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)


    def _draw_waypoints(self, world, start_loc, end_loc):
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
        

    def run_step(self, debug=False):
        if not self._agent:
            self._agent = BasicAgent(self._vehicle, target_speed=self._target_speed, opt_dict={'base_tlight_threshold': 7.0})

            plan = []

            for i in range(len(self._waypoints)-1):
                start_loc = carla.Location(self._waypoints[i][0], self._waypoints[i][1], self._waypoints[i][2])
                end_loc = carla.Location(self._waypoints[i+1][0], self._waypoints[i+1][1], self._waypoints[i+1][2])
                for waypoint, road_option in self._global_planner.trace_route(start_loc, end_loc):
                    plan.append((waypoint, road_option))
               

            self.set_global_plan(plan)

            self._agent.set_global_plan(plan)

            return carla.VehicleControl()
        
        else:
            """
            # TODO: perfrom lane change when vehicle detected
            if self._vehicle_obstacle_detected()[0]:
                print("Vehicle Detected")

                
                waypoint = self._map.get_waypoint(self._vehicle.get_location())
                lane_change = waypoint.lane_change

                if lane_change == carla.LaneChange.NONE:
                    print("None")
                elif lane_change == carla.LaneChange.Right:
                    print("Right")
                    self._agent.lane_change(direction='right', lane_change_time=1)
                elif lane_change == carla.LaneChange.Left:
                    print("Right")
                    self._agent.lane_change(direction='left', lane_change_time=1)
                elif lane_change == carla.LaneChange.Both:
                    directions = ['left', 'right']
                    direction = random.choice(directions)
                    self._agent.lane_change(direction=direction, lane_change_time=1)

                return self._agent.run_step()
            """
            

            return self._agent.run_step()