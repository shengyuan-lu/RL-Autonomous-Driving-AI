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

    INIT_DIST_BETWEEN_VEHICLES = 10.0

    def __init__(self):
        # connect to carla server
        self.client = Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        # self.client.load_world('Town05')
        self.render_mode = 'human'

        self.action_space = None
        self.observation_space = None

        self.learning_vehicle = None  # learning agent
        self.privilege_vehicle = None  # privileged agent
        self.privilege_agent = None
        self.collision_sensor = None
        self.camera_sensor = None
        self.lane_invasion_sensor = None
        self.spawn_point = None
        self.pv_spawn_point = None
        self.camera_data = None
        self.lane_invasion_data = None
        self.collision_data = None
        self.episode_start = None
        self.obstacles = []

        self._init_observation_action_space()  # initialize observation space
        self._setup_sensors_and_actors()
        
        # self._generate_obstacles_ahead(20)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _init_observation_action_space(self):
        # observation space with camera image only
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 3), dtype=np.uint8)

        # FIXME: currently set brake to 0.0 so vehicle does not stop initially
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0], dtype=np.float32))  # throttle, steer, brake
    

    def _setup_sensors_and_actors(self):
        # setup basic agent
        if not self.privilege_vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            self.pv_spawn_point = self.world.get_map().get_spawn_points()[0]
            self.pv_spawn_point.location.x -= 10.0
            self.privilege_vehicle = self.world.spawn_actor(vehicle_bp, self.pv_spawn_point)
            # wrap it with basic agent
            self.privilege_agent = BasicAgent(self.privilege_vehicle, opt_dict={'ignore_traffic_lights': True, 'target_speed': 20})
            # end_location = carla.Location(x=-190.971420, y=-68.036812, z=0.0)
            # self.privilege_agent.set_destination(end_location, start_location=self.pv_spawn_point)

        # setup learning agent
        if not self.learning_vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.cybertruck')  # get specific vehicle
            self.spawn_point = self.world.get_map().get_spawn_points()[0]
            self.spawn_point.location.x = (self.pv_spawn_point.location.x - self.INIT_DIST_BETWEEN_VEHICLES)
            self.learning_vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
            # if self.vehicle == None:
            #     spawn_points = self.world.get_map().get_spawn_points()
            #     self.spawn_point = random.choice(spawn_points)
            #     print(self.spawn_point)
            #     self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        
        # Setup sensors
        if not self.collision_sensor:
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.learning_vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        if not self.camera_sensor:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.CAMERA_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.CAMERA_HEIGHT))
            camera_bp.set_attribute('fov', str(self.FOV))
            self.camera_sensor = self.world.spawn_actor(camera_bp, 
                                                        carla.Transform(carla.Location(x=3.0, z=10.0), carla.Rotation(pitch=-90)),
                                                        attach_to=self.learning_vehicle)
            self.camera_sensor.listen(lambda data: self.process_image(data))
        
        if not self.lane_invasion_sensor:
            lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.learning_vehicle)
            self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _generate_obstacles(self, size):
        # randomly generate obstacles within specific distance to the agent
        self.obstacles = []
        vehicle_location = self.learning_vehicle.get_location()

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
        vehicle_transform = self.learning_vehicle.get_transform()

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
    
    def process_image(self, img):
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

    def _is_same_lane(self, vehicle1, vehicle2):
        waypoint1 = self.world.get_map().get_waypoint(vehicle1.get_location())
        waypoint2 = self.world.get_map().get_waypoint(vehicle2.get_location())

        same_lanes = waypoint1.lane_id == waypoint2.lane_id

        return same_lanes

    def get_observation(self):
        # get camera image
        observation = self.camera_data

        # # get the transform of the vehicle and velocity of the vehicle
        # privilege_vehicle_transform = self.privilege_vehicle.get_transform()
        # privilege_vehicle_velocity = self.privilege_vehicle.get_velocity()

        # learning_vehicle_transform = self.learning_vehicle.get_transform()
        # learning_vehicle_velocity = self.learning_vehicle.get_velocity()

        # relative_pos = np.array([privilege_vehicle_transform.location.x - learning_vehicle_transform.location.x,
        #                         privilege_vehicle_transform.location.y - learning_vehicle_transform.location.y,
        #                         privilege_vehicle_transform.location.z - learning_vehicle_transform.location.z])
        
        # relative_vel = np.array([privilege_vehicle_velocity.x - learning_vehicle_velocity.x,
        #                         privilege_vehicle_velocity.y - learning_vehicle_velocity.y,
        #                         privilege_vehicle_velocity.z - learning_vehicle_velocity.z])
        
        
        # speed = np.sqrt(np.square(learning_vehicle_velocity.x) + np.square(learning_vehicle_velocity.y) + np.square(learning_vehicle_velocity.z))

        return observation

    def reset(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
            self.camera_sensor = None
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.privilege_vehicle:
            self.privilege_vehicle.destroy()
            self.privilege_vehicle = None
            self.privilege_agent = None
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
            self.lane_invasion_sensor = None
        if self.learning_vehicle:
            self.learning_vehicle.destroy()
            self.learning_vehicle = None

        self._setup_sensors_and_actors()

        self.world.tick()

        self.episode_start = self.world.get_snapshot().timestamp.elapsed_seconds
        self.collision_data = False
        self.lane_invasion_data = False

        obs = self.get_observation()
        return obs  # return initial observation

    def _reward_for_reaching_waypoint(self, learning_vehicle, privilege_vehicle, close_threshold=2.0):   
        reward = 0.0

        learning_vehicle_wp = self.world.get_map().get_waypoint(learning_vehicle.get_location())
        privilege_vehicle_wp = self.world.get_map().get_waypoint(privilege_vehicle.get_location())

        distance_to_waypoint = learning_vehicle_wp.transform.location.distance(privilege_vehicle_wp.transform.location)
        #print(current_wp)
        print(distance_to_waypoint)
        if distance_to_waypoint <= close_threshold:
            print("reached to waypoint")
            reward += 1.0
        else:
            reward -= 1.0

        return reward
        

    
    def step(self, action):
        # step priviledge agent
        control = self.privilege_agent.run_step()
        self.privilege_vehicle.apply_control(control)

        # step learning agent
        throttle, steer, brake = action  # action with two items
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.learning_vehicle.apply_control(control)


        self.world.tick()

        done = False
        reward = 0.0

        ## define rewards
    
        # calculate distance between privilege and learning vehicles
        privilege_vehicle_transform = self.privilege_vehicle.get_transform()
        learning_vehicle_transform = self.learning_vehicle.get_transform()
        distance = np.sqrt((privilege_vehicle_transform.location.x - learning_vehicle_transform.location.x) ** 2 +
                            (privilege_vehicle_transform.location.y - learning_vehicle_transform.location.y) ** 2 +
                            (privilege_vehicle_transform.location.z - learning_vehicle_transform.location.z) ** 2)

        # assign reward/penalty based on distance
        if distance >= 2 and distance <= 10:
            reward += 1.0
        elif distance > 10 and distance <= 15:
            reward += 0.5

        # reward for being in the same lane
        if self._is_same_lane(self.privilege_vehicle, self.learning_vehicle):
            reward += 1.0
        else:
            reward -= 5.0

        # FIXME: reward for reaching the waypoint
        # reward += self._reward_for_reaching_waypoint(self.learning_vehicle, self.privilege_vehicle)
        
        # # steer penalty
        # if abs(steer) > 0.5:
        #     reward -= abs(steer) * 0.5

        # lane invasion penalty
        
        # if self.lane_invasion_data:
        #     print("lane invasion")
        #     reward -= 1.0

        # done once collision occurs
        if self.collision_data or distance > 15 or distance < 2:
            done = True
            reward = -300.0
        
        # done if timeout
        if self.world.get_snapshot().timestamp.elapsed_seconds - self.episode_start > 50:
            print("Time out")
            done = True

        # no collision, reward for current frame
        if not done:
            reward += 0.1

        # get observation
        obs = self.get_observation()

        info = {}

        if self.render_mode == 'human':
            self.render()

        return obs, reward, done, info

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
        if self.privilege_vehicle:
            self.privilege_vehicle.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()
        if self.learning_vehicle:
            self.learning_vehicle.destroy()
        

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
        
