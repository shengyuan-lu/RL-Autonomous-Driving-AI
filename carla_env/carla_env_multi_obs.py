import gym
import carla

import random
import cv2
import numpy as np
from carla import Client

from agents.template_agents.behavior_agent import BehaviorAgent

class CarlaEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    CAMERA_WIDTH = int(512 / 2)
    CAMERA_HEIGHT = int(512 / 2)
    FOV = int(90)
    PRIVILEGE_START_LOC = carla.Location(x=5, y=-200, z=1.0)

    INIT_DIST_BETWEEN_VEHICLES = 7.0

    def __init__(self):
        # connect to carla server
        self.client = Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.render_mode = 'human'

        self.action_space = None
        self.observation_space = None

        self.traffic_manager = None
        self.learning_vehicle = None  # learning agent
        self.privilege_vehicle = None  # priviledge agent
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
        self._init_traffic_manager()  # initialize traffic manager
        self._setup_sensors_and_actors()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _init_traffic_manager(self):
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

    def _init_observation_action_space(self):
        camera_image_space = gym.spaces.Box(low=0, high=255, shape=(self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 3),
                                            dtype=np.uint8)
        telemetry_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0])  # relative position, relative velocity, speed
        telemetry_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])

        telemtry_space = gym.spaces.Box(low=telemetry_low, high=telemetry_high, dtype=np.float32)

        # observation space contains both camera image and telemetry data
        self.observation_space = gym.spaces.Dict({
            'camera': camera_image_space,
            'telemetry': telemtry_space,
        })

        # FIXME: currently set brake to 0.0 so vehicle does not stop initially
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0, 0.0]),
                                           high=np.array([1.0, 1.0, 1.0], dtype=np.float32))  # throttle, steer, brake

    def _setup_sensors_and_actors(self):
        # setup privilege agent to start location
        if not self.privilege_vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
            self.pv_spawn_point = carla.Transform(self.PRIVILEGE_START_LOC, carla.Rotation(yaw=180.0))
            # self.pv_spawn_point.location.x -= 10.0
            self.privilege_vehicle = self.world.spawn_actor(vehicle_bp, self.pv_spawn_point)
            self.privilege_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            # wrap it with behavior agent
            self.privilege_agent = BehaviorAgent(self.privilege_vehicle, behavior='normal')

            # Set the agent destination
            spawn_points = self.world.get_map().get_spawn_points()
            destination = random.choice(spawn_points).location
            self.privilege_agent.set_destination(destination)

        # setup learning agent
        if not self.learning_vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.cybertruck')  # get specific vehicle
            self.spawn_point = carla.Transform(
                carla.Location(self.PRIVILEGE_START_LOC.x + self.INIT_DIST_BETWEEN_VEHICLES, self.PRIVILEGE_START_LOC.y,
                               self.PRIVILEGE_START_LOC.z),
                carla.Rotation(yaw=180.0))
            self.learning_vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)

        # Setup sensors
        if not self.collision_sensor:
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(),
                                                           attach_to=self.learning_vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        if not self.camera_sensor:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
            camera_bp.set_attribute('image_size_x', str(self.CAMERA_WIDTH))
            camera_bp.set_attribute('image_size_y', str(self.CAMERA_HEIGHT))
            camera_bp.set_attribute('fov', str(self.FOV))
            self.camera_sensor = self.world.spawn_actor(camera_bp,
                                                        carla.Transform(carla.Location(x=3.0, z=10.0),
                                                                        carla.Rotation(pitch=-90)),
                                                        attach_to=self.learning_vehicle)
            self.camera_sensor.listen(lambda data: self.process_image(data))

        if not self.lane_invasion_sensor:
            lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(),
                                                               attach_to=self.learning_vehicle)
            self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def process_image(self, img):
        img.convert(carla.ColorConverter.CityScapesPalette)  # convert to raw data
        bgra = np.array(img.raw_data).reshape((self.CAMERA_WIDTH, self.CAMERA_HEIGHT, 4))
        bgr = bgra[:, :, :3]

        ROAD_COLOR_BGR = [128, 64, 128]
        VEHICLE_COLOR_BGR = [142, 0, 0]
        ROAD_LINE_COLOR_BGR = [50, 234, 157]

        # create mask based on color
        road_mask = np.all(bgr == ROAD_COLOR_BGR, axis=-1)
        vehicle_mask = np.all(bgr == VEHICLE_COLOR_BGR, axis=-1)
        road_line_mask = np.all(bgr == ROAD_LINE_COLOR_BGR, axis=-1)
        combined_mask = road_mask | vehicle_mask | road_line_mask

        processed_img = np.zeros_like(bgr)
        processed_img[combined_mask] = bgr[combined_mask]

        self.camera_data = processed_img

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
        img_data = self.camera_data

        # get the transform of the vehicle and velocity of the vehicle
        privilege_vehicle_transform = self.privilege_vehicle.get_transform()
        privilege_vehicle_velocity = self.privilege_vehicle.get_velocity()

        learning_vehicle_transform = self.learning_vehicle.get_transform()
        learning_vehicle_velocity = self.learning_vehicle.get_velocity()

        relative_pos = np.array([privilege_vehicle_transform.location.x - learning_vehicle_transform.location.x,
                                 privilege_vehicle_transform.location.y - learning_vehicle_transform.location.y])

        relative_vel = np.array([privilege_vehicle_velocity.x - learning_vehicle_velocity.x,
                                 privilege_vehicle_velocity.y - learning_vehicle_velocity.y])

        speed = np.sqrt(np.square(learning_vehicle_velocity.x) + np.square(learning_vehicle_velocity.y))

        observation = {
            'camera': img_data,
            'telemetry': np.concatenate([relative_pos, relative_vel, [speed]])
        }

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

        # define rewards

        # calculate distance between privilege and learning vehicles
        privilege_vehicle_transform = self.privilege_vehicle.get_transform()
        learning_vehicle_transform = self.learning_vehicle.get_transform()
        distance = np.sqrt((privilege_vehicle_transform.location.x - learning_vehicle_transform.location.x) ** 2 +
                           (privilege_vehicle_transform.location.y - learning_vehicle_transform.location.y) ** 2)

        # assign reward/penalty based on distance
        if 2 <= distance <= 10:
            reward += 1.0
        elif 10 < distance <= 15:
            reward += 0.5

        # reward for being in the same lane
        if self._is_same_lane(self.privilege_vehicle, self.learning_vehicle):
            reward += 1.0
        else:
            reward -= 5.0

        # steer penalty
        if abs(steer) > 0.5:
            reward -= abs(steer) * 0.5

        # speed penalty when greater than previlege vehicle
        privilege_vehicle_velocity = self.privilege_vehicle.get_velocity()
        learning_vehicle_velocity = self.learning_vehicle.get_velocity()
        privilege_vehicle_speed = np.sqrt(
            np.square(privilege_vehicle_velocity.x) + np.square(privilege_vehicle_velocity.y))
        learning_vehicle_speed = np.sqrt(
            np.square(learning_vehicle_velocity.x) + np.square(learning_vehicle_velocity.y))
        if learning_vehicle_speed > privilege_vehicle_speed:
            reward -= 1.0
        else:
            reward += 1.0

        # lane invasion penalty
        if self.lane_invasion_data:
            reward -= 1.0

        # Maintain position near the center of the current lane
        learning_vehicle_location = learning_vehicle_transform.location
        current_waypoint = self.world.get_map().get_waypoint(learning_vehicle_location)
        lane_width = current_waypoint.lane_width
        current_lane_center = current_waypoint.transform.location
        current_lane_center.x += lane_width / 2
        lateral_deviation = abs(learning_vehicle_transform.location.y - current_lane_center.y)
        reward += 1.0 / (lateral_deviation + 1)

        # done once collision occurs
        if self.collision_data or distance > 15 or distance < 2:
            done = True
            reward = -200.0

        # done if timeout
        if self.world.get_snapshot().timestamp.elapsed_seconds - self.episode_start > 100:
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
        # our_agents = actor_list.filter('vehicle.*')

        # for agent in our_agents:
        #     if agent.id != self.vehicle.id:
        #         print(f"Destroying agent: {agent.id}")
        #         agent.destroy()

        # destroy obstacles
        for obstacle in self.obstacles:
            # print(f"Destroying obstacle: {obstacle.id}")
            obstacle.destroy()
