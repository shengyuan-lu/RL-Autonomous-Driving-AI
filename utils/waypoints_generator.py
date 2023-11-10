import carla
import pygame
from pygame.locals import *
import json
import csv

from util.actor_movement_recorder import MovementRecorder


# connect to simulator
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# initialize world
world = client.get_world()
client.load_world('Town05')

# Get the blueprint for the car
# Get the blueprint library and filter for the vehicle blueprints
vehicle_bps = world.get_blueprint_library().filter('*vehicle*')  # get list of vehicles
vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')  # get specific vehicle
vehicle_bp.set_attribute('role_name', 'hero')  # set player's vehicle blueprint as hero

car_transform = world.get_map().get_spawn_points()[0]

car = world.spawn_actor(vehicle_bp, car_transform)

mr = MovementRecorder(car)

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Control')
clock = pygame.time.Clock()


def run():
    _flag = True
    _movement_record = False
    _start_time = world.get_snapshot().timestamp.elapsed_seconds

    while _flag:
        clock.tick(60)

        # Get current timestamp and calculate elapsed time
        timestamp = world.get_snapshot().timestamp.elapsed_seconds
        elapsed_time = timestamp - _start_time

        # record location of actor every 3 secs
        if _movement_record & (elapsed_time >= 1):
            mr.record_movement()
            _start_time = world.get_snapshot().timestamp.elapsed_seconds

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.key == K_b:
                    _movement_record = True
                    start_loc = car.get_location()
                    print("Begin tracking")
                elif event.key == K_p:
                    print("End tracking")
                    end_loc = car.get_location()
                    print(end_loc)
                    _flag = False
                    break
                elif event.key == K_q:
                    control = carla.VehicleControl(brake=1.0, steer=0.0)
                    car.apply_control(control)
                elif event.key == K_UP:
                    control = carla.VehicleControl(throttle=1.0, steer=0.0)
                    car.apply_control(control)
                elif event.key == K_DOWN:
                    control = carla.VehicleControl(throttle=1.0, steer=0.0, reverse=True)
                    car.apply_control(control)
                elif event.key == K_LEFT:
                    reverse = False
                    if event.key == K_DOWN:
                        reverse = True
                    control = carla.VehicleControl(throttle=0, steer=-0.5, reverse=reverse)
                    car.apply_control(control)
                elif event.key == K_RIGHT:
                    reverse = False
                    if event.key == K_DOWN:
                        reverse = True
                    control = carla.VehicleControl(throttle=0, steer=0.5, reverse=reverse)
                    car.apply_control(control)
                else:
                    control = carla.VehicleControl()
                    car.apply_control(control)

        # Update the Pygame display
        pygame.display.flip() 

    pygame.quit()
    car.destroy()

    



    #draw_waypoints(world, start_loc, end_loc)
    
    plan = dict()
    if 'route_1' not in plan:
        plan['route_1'] = dict()

        if 'start_loc' not in plan['route_1']:
            plan['route_1']['start_loc'] = (start_loc.x, start_loc.y, start_loc.z)
        
        if 'end_loc' not in plan['route_1']:
            plan['route_1']['end_loc'] = (end_loc.x, end_loc.y, end_loc.z)


    print(plan)
    with open("plan.json", "w") as f:
        json.dump(plan, f)

    movements_data = mr.get_movements_data()
    print(movements_data)

    for i in range(len(movements_data) - 1):
        start = carla.Location(movements_data[i][0], movements_data[i][1], movements_data[i][2])
        end = carla.Location(movements_data[i+1][0], movements_data[i+1][1], movements_data[i+1][2])
        

    with open("movements_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(movements_data)
        

if __name__ == "__main__":
    run()


