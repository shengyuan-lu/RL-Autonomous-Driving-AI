# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

class DetourAgent(BasicAgent):
    def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
        self._vehicle = vehicle
        self._destination = destination
        self._basic_agent = BasicAgent(self._vehicle)
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._start_location = self._vehicle.get_location()
        self._detour_route = None
    
    def _setup_lidar_sensor(self):
        # Set up LiDAR sensor to detect obstacles
        blueprint_library = self._world.get_blueprint_library()
        lidar_blueprint = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute('range', str(self._lidar_range))
        lidar_blueprint.set_attribute('sensor_tick', '0.1')
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self._lidar_sensor = self._world.spawn_actor(lidar_blueprint, transform, attach_to=self._vehicle)
        self._lidar_sensor.listen(lambda data: self._process_lidar_data(data))
    
    def _process_lidar_data(self, data):
        # Check LiDAR sensor data for obstacles within the sensor range
        for detection in data:
            if detection.depth <= self._lidar_range and detection.object_tag == 'Static':
                # Obstacle detected
                self._obstacle_detected = True
    
    def _detect_obstacle(self):
        # Check if an obstacle is currently detected by the LiDAR sensor
        if self._lidar_sensor is None:
            self._setup_lidar_sensor()
        return self._obstacle_detected
    
    def _navigate_to_safe_point(self):
        # Use the BasicAgent to navigate to a safe point (e.g. the start of the detour route)
        safe_point = self._start_location
        self._basic_agent.set_destination(safe_point)
        self._basic_agent.run_step()
    
    def _plan_detour_route(self):
        # Use the map data to plan a detour route around the blocked area
        current_location = self._vehicle.get_location()
        start_waypoint = self._map.get_waypoint(current_location)
        end_waypoint = self._map.get_waypoint(self._destination.location)
        # Implement a detour planning algorithm to find a new route that avoids the blocked area
        self._detour_route = self._map.get_random_waypoint()
    
    def run_step(self):
        if self._detour_route is None:
            if self._detect_obstacle():
                self._navigate_to_safe_point()
                self._plan_detour_route()
            else:
                # No obstacle detected, follow the original route
                self._basic_agent.set_destination(self._destination)
        else:
            # Follow the detour route until the blocked area is passed
            current_location = self._vehicle.get_location()
            current_waypoint = self._map.get_waypoint(current_location)
            if current_waypoint.road_id == self._detour_route.road_id:
                # We're on the detour route, follow it
                self._basic_agent.set_destination(self._detour_route)
            else:
                # We've passed the blocked area, resume the original route
                self._basic_agent.set_destination(self._destination)
                self._detour_route = None
        # Run the BasicAgent
        self._basic_agent.run_step()