import gym
import carla
import math
import numpy as np
from carla import Client
from carla import VehicleControl
from carla_env.carla_sync_mode import CarlaSyncMode
#from leaderboard.autoagents.detour_agents.my_detour_agent import DetourAgent

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        # connect to carla server
        self.client = Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # initialize autopilot
        #self.agent = DetourAgent(self.vehicle)

        # specify gym environment action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0], dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8)  # use 300x200 RGB image for observation space

        self.vehicle = None
        self.obstacles = []
        self.collision_sensor = None
        self.camera_sensor = None
        self.spawn_point = None
        self.camera_data = None
        self.collision_data = None
        self.episode_start = None

        self.setup_sensors_and_actors()
        self._generate_obstacles_ahead(15)


    def setup_sensors_and_actors(self):
        if not self.vehicle:
            vehicle_bp = self.world.get_blueprint_library().find('vehicle.tesla.cybertruck')  # get specific vehicle
            self.spawn_point = self.world.get_map().get_spawn_points()[1]
            self.vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        
        # Setup sensors
        if not self.collision_sensor:
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        if not self.camera_sensor:
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '512')
            camera_bp.set_attribute('image_size_y', '512')
            camera_bp.set_attribute('fov', '100')
            self.camera_sensor = self.world.spawn_actor(camera_bp, 
                                                        carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                                                        attach_to=self.vehicle)
            self.camera_sensor.listen(lambda data: self._get_pixel_obs(data))

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
        bgra = np.array(img.raw_data).reshape((512, 512, 4))
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        self.camera_data = rgb
    
    def _on_collision(self, event):
        self.collision_data = event

    def reset(self):
        self.vehicle.set_transform(self.spawn_point)
        self.episode_start = self.world.get_snapshot().timestamp.elapsed_seconds
        self.collision_data = None
        return self.camera_data  # return initial observation
    
    def step(self, action):
        throttle, steer = action  # action with two items
        control = carla.VehicleControl(throttle=float(throttle), steer=float(steer))
        self.vehicle.apply_control(control)

        self.world.tick()

        done = False
        reward = 0.0

        # done once collision occurs
        if self.collision_data:
            done = True
            reward = -100.0
        
        # no collision, reward for current frame
        if not done:
            reward = 1.0

        info = {}

        return self.camera_data, reward, done, info


    def render(self, mode='human'):
        print(f'Step: {self.world.get_snapshot().timestamp.elapsed_seconds - self.episode_start:.2f}s')

    def close(self):
        if self.camera_sensor:
            self.camera_sensor.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()

        # destroy all other vehicle actors
        actor_list = self.world.get_actors()
        agents = actor_list.filter('vehicle.*')

        for agent in agents:
            if agent.id != self.vehicle.id:
                print(f"Destroying agent: {agent.id}")
                agent.destroy()

        # destroy obstacles
        for obstacle in self.obstacles:
            obstacle.destroy()
        
