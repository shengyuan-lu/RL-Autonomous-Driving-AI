import gym
import carla
import time
import math
import random
import cv2
import numpy as np
from carla import Client
from carla import VehicleControl
#from leaderboard.autoagents.detour_agents.my_detour_agent import DetourAgent
from agents.navigation.basic_agent import BasicAgent

class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    CAMERA_WIDTH = int(512 / 2)
    CAMERA_HEIGHT = int(512 / 2)
    FOV = int(90)

    def __init__(self):
        # connect to carla server
        self.client = Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # initialize autopilot
        #self.agent = DetourAgent(self.vehicle)

        # specify gym environment action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 3), dtype=np.uint8)  # use 300x200 RGB image for observation space

        self.vehicle = None
        self.privilege_vehicle = None
        self.obstacles = []
        self.collision_sensor = None
        self.camera_sensor = None
        self.lane_invasion_sensor = None
        self.spawn_point = None
        self.pv_spawn_point = None
        self.camera_data = None
        self.lane_invasion_data = None
        self.collision_data = None
        self.episode_start = None

        self.setup_sensors_and_actors()
        # self._generate_obstacles_ahead(20)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def setup_sensors_and_actors(self):
        # setup basic agent
        if not self.privilege_vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            self.pv_spawn_point = self.world.get_map().get_spawn_points()[0]
            self.privilege_vehicle = self.world.spawn_actor(vehicle_bp, self.pv_spawn_point)
            # wrap it with basic agent
            self.priviledge_agent = BasicAgent(self.privilege_vehicle)
        

        # setup learning agent
        if not self.vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.cybertruck')  # get specific vehicle
            self.spawn_point = self.world.get_map().get_spawn_points()[0]
            self.spawn_point.location.x -= 10.0
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
            # if self.vehicle == None:
            #     spawn_points = self.world.get_map().get_spawn_points()
            #     self.spawn_point = random.choice(spawn_points)
            #     print(self.spawn_point)
            #     self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        
        # Setup sensors
        if not self.collision_sensor:
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        if not self.camera_sensor:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.CAMERA_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.CAMERA_HEIGHT))
            camera_bp.set_attribute('fov', str(self.FOV))
            self.camera_sensor = self.world.spawn_actor(camera_bp, 
                                                        carla.Transform(carla.Location(x=3.0, z=15.0), carla.Rotation(pitch=-90)),
                                                        attach_to=self.vehicle)
            self.camera_sensor.listen(lambda data: self._get_pixel_obs(data))
        
        if not self.lane_invasion_sensor:
            lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
            self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _generate_obstacles(self, size):
        # randomly generate obstacles within specific distance to the agent
        self.obstacles = []
        vehicle_location = self.vehicle.get_location()

        for _ in range(size):
            distance_range = [10, 50]

            distance = np.random.uniform(*distance_range)
            angle = np.random.uniform(0, 360)  # spawn obstacles straight ahead
            radian_angle = np.radians(angle)

            obstacle_location = carla.Location(
            x = vehicle_location.x + distance * math.cos(radian_angle),
            y = vehicle_location.y + distance * math.sin(radian_angle),
            z = vehicle_location.z)

            blueprint_library = self.world.get_blueprint_library()
            obstacle_bp = np.random.choice(blueprint_library.filter('static.prop.trafficcone01'))

            obstacle = self.world.try_spawn_actor(obstacle_bp, carla.Transform(obstacle_location))
            if obstacle is not None:
                self.obstacles.append(obstacle)
    
    def _generate_obstacles_ahead(self, size):
        vehicle_transform = self.vehicle.get_transform()

        for _ in range(size):
            distance_range = [10, 20]
            distance = np.random.uniform(*distance_range)

            angle = np.random.uniform(-90, 90) 
            radian_angle = np.radians(angle)

            forward_vec = vehicle_transform.get_forward_vector()
            spread_vec = carla.Vector3D(
                x = forward_vec.x * math.cos(radian_angle) - forward_vec.y * math.sin(radian_angle),
                y = forward_vec.x * math.sin(radian_angle) + forward_vec.y * math.cos(radian_angle),
                z = forward_vec.z)
            
            spread_vec *= distance
            
            spawn_location = vehicle_transform.location + carla.Location(spread_vec)
            spawn_location.z = vehicle_transform.location.z

            blueprint_library = self.world.get_blueprint_library()
            obstacle_bp = np.random.choice(blueprint_library.filter('static.prop.trafficcone01'))
            obstacle_transform = carla.Transform(spawn_location, carla.Rotation())

            obstacle = self.world.try_spawn_actor(obstacle_bp, obstacle_transform)
            if obstacle is not None:
                self.obstacles.append(obstacle)
    
    def _get_pixel_obs(self, img):
        img.convert(carla.ColorConverter.CityScapesPalette)  # convert to raw data
        bgra = np.array(img.raw_data).reshape((self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 4))
        bgr = bgra[:, :, :3]
        # rgb = np.flip(bgr, axis=2)
        # self.camera_data = rgb
        self.camera_data = bgr
    
    def _on_collision(self, event):
        self.collision_data = event
    
    def _on_lane_invasion(self, event):
        self.lane_invasion_data = event

    def _get_vehicles_locations(self):
        priviledge_vehicle_location = self.privilege_vehicle.get_location()
        learning_vehicle_location = self.vehicle.get_location()

        return priviledge_vehicle_location, learning_vehicle_location

    def reset(self):
        # reset previledge agent
        self.privilege_vehicle.set_transform(self.pv_spawn_point)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))  # reset velocity and steering
        time.sleep(0.5)  # wait for 1 second
        self.vehicle.set_transform(self.spawn_point)
        self.episode_start = self.world.get_snapshot().timestamp.elapsed_seconds
        self.collision_data = None
        return self.camera_data  # return initial observation
    
    def step(self, action):
        # step priviledge agent
        control = self.priviledge_agent.run_step()
        self.privilege_vehicle.apply_control(control)

        # step learning agent
        throttle, steer = action  # action with two items
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)


        self.world.tick()

        done = False
        reward = 0.0

        ## define rewards
        # done once collision occurs
        if self.collision_data:
            done = True
            reward = -50.0

        # steer penalty
        if abs(steer) > 0.5:
            reward -= abs(steer)

        # # lane invasion penalty
        # if self.lane_invasion_data:
        #     print("lane invasion")
        #     reward -= 5.0

        # no collision, reward for current frame
        if not done:
            reward += 1.0

        info = {}

        return self.camera_data, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            img = self.camera_data
            cv2.imshow('Camera', img)
            cv2.waitKey(1)
        
    def close(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.priviledge_agent:
            self.priviledge_agent.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        

        # # destroy all other vehicle actors
        # actor_list = self.world.get_actors()
        # agents = actor_list.filter('vehicle.*')

        # for agent in agents:
        #     if agent.id != self.vehicle.id:
        #         print(f"Destroying agent: {agent.id}")
        #         agent.destroy()

        # destroy obstacles
        for obstacle in self.obstacles:
            # print(f"Destroying obstacle: {obstacle.id}")
            obstacle.destroy()
        
