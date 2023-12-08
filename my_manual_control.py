#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change camera position

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit

    B            : print locations
    T            : teleport to egos
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import os
import sys
import time
import collections
import datetime
import logging
import math
import weakref
import sys

import xml.etree.ElementTree as ET

try:
    import pygame
    import pygame.gfxdraw
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_w
    from pygame.locals import K_a
    from pygame.locals import K_s
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_m
    from pygame.locals import K_COMMA
    from pygame.locals import K_PERIOD
    from pygame.locals import K_p
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_r
    from pygame.locals import K_b
    from pygame.locals import K_t
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):

    restarted = False

    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0


    def restart(self):

        if self.restarted:
            return
        self.restarted = True

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        # while self.player is None:
        #     print("Waiting for the ego vehicle...")
        #     time.sleep(1)
        #     possible_vehicles = self.world.get_actors().filter('vehicle.lincoln.*')
        #     for vehicle in possible_vehicles:
        #         if vehicle.attributes['role_name'] == 'hero':
        #             print("Ego vehicle found")
        #             self.player = vehicle
        #             break

        print("Spawning ego vehicle...")
        PRIVILEGE_START_LOC = carla.Location(x=5, y=-200, z=1.0)
        pv_spawn_point = carla.Transform(PRIVILEGE_START_LOC, carla.Rotation(yaw=180.0))
        vehicle_blueprint = np.random.choice(self.world.get_blueprint_library().filter('vehicle.lincoln.*'))
        vehicle_blueprint.set_attribute('role_name', 'hero')
        
        if pv_spawn_point:
            self.player = self.world.try_spawn_actor(vehicle_blueprint, pv_spawn_point)
            if self.player:
                print("Ego vehicle spawned.")
            else:
                print("Failed to spawn ego vehicle.")
        else:
            print("No spawn points available.")
        
        self.player_name = self.player.type_id

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        # self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        # self.gnss_sensor = GnssSensor(self.player)
        # self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.world.wait_for_tick()

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            return False

        self.hud.tick(self, clock)
        return True

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        self.hud.render_gauge(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor]
            # self.lane_invasion_sensor.sensor,
            # self.gnss_sensor.sensor,
            # self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        # if self.player is not None:
        #     self.player.destroy()
        # write history waypoints, for route generation
        # waypoints = ET.Element('waypoints')
        # for wp in self.hud._passed_waypoints:
        #     waypoint = ET.SubElement(waypoints, 'waypoint')
        #     waypoint.set('pitch', '0.0')
        #     waypoint.set('roll', '0.0')
        #     waypoint.set('yaw', '{}'.format(wp.transform.rotation.yaw))
        #     waypoint.set('x', '{}'.format(wp.transform.location.x))
        #     waypoint.set('y', '{}'.format(wp.transform.location.y))
        #     waypoint.set('z', '{}'.format(wp.transform.location.z))
        # # wp = self.hud._passed_waypoints[-1]
        # # waypoint = ET.SubElement(waypoints, 'waypoint')
        # # waypoint.set('pitch', '0.0')
        # # waypoint.set('roll', '0.0')
        # # waypoint.set('yaw', '{}'.format(wp.transform.rotation.yaw))
        # # waypoint.set('x', '{}'.format(wp.transform.location.x))
        # # waypoint.set('y', '{}'.format(wp.transform.location.y))
        # # waypoint.set('z', '{}'.format(wp.transform.location.z))
        # data_string = ET.tostring(waypoints, encoding='unicode')
        # with open('_out/manual_passed_waypoints.xml', 'w') as _f:
        #     _f.write(data_string)


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self._steer_cache = 0.0
        world.player.set_autopilot(self._autopilot_enabled)
        world.player.set_light_state(self._lights)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    

        self.current_ego_idx = 0

    def parse_events(self, client, world, clock):
        current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.player.get_control().gear
                    world.hud.notification('%s Transmission' %
                                            ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                    current_lights ^= carla.VehicleLightState.Special1
                elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                    current_lights ^= carla.VehicleLightState.HighBeam
                elif event.key == K_l:
                    # Use 'L' key to switch between lights:
                    # closed -> position -> low beam -> fog
                    if not self._lights & carla.VehicleLightState.Position:
                        world.hud.notification("Position lights")
                        current_lights |= carla.VehicleLightState.Position
                    else:
                        world.hud.notification("Low beam lights")
                        current_lights |= carla.VehicleLightState.LowBeam
                    if self._lights & carla.VehicleLightState.LowBeam:
                        world.hud.notification("Fog lights")
                        current_lights |= carla.VehicleLightState.Fog
                    if self._lights & carla.VehicleLightState.Fog:
                        world.hud.notification("Lights off")
                        current_lights ^= carla.VehicleLightState.Position
                        current_lights ^= carla.VehicleLightState.LowBeam
                        current_lights ^= carla.VehicleLightState.Fog
                elif event.key == K_i:
                    current_lights ^= carla.VehicleLightState.Interior
                elif event.key == K_z:
                    current_lights ^= carla.VehicleLightState.LeftBlinker
                elif event.key == K_x:
                    current_lights ^= carla.VehicleLightState.RightBlinker
                elif event.key == K_b:
                    print(world.player.get_transform())
                    sys.stdout.flush()
                elif event.key == K_t:
                    vehicles = world.world.get_actors().filter('vehicle.*')

                    ego_vehicles = []
                    for vehicle in vehicles:
                        if "tesla.cybertruck" in vehicle.type_id \
                            and vehicle.get_transform().location.z > -1:  # Check if the vehicle is a Tesla Cybertruck
                                ego_vehicles.append(vehicle)
                    try:
                        ego_vehicle = ego_vehicles[self.current_ego_idx]
                        tele_transform = ego_vehicle.get_transform()
                        tele_transform.location -= 10 * tele_transform.get_forward_vector()
                        tele_transform.location.z += 0.2
                        world.player.set_transform(tele_transform)
                        self.current_ego_idx += 1

                        if self.current_ego_idx >= len(ego_vehicles):
                            self.current_ego_idx = 0

                    except IndexError:
                        pass

        if not self._autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else: # Remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else: # Remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights: # Change the light state only if necessary
                self._lights = current_lights
                world.player.set_light_state(carla.VehicleLightState(self._lights))
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_speedToSound(self, world, clock):
        # play sound
        v = world.player.get_velocity()
        speed = (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        self.car_sound = pygame.mixer.Sound(self.car_sound_array_idle)
        self.car_sound.set_volume(0.6)
        if self.sound_played_time > 2e2:
            # every 0.5 seconds, reset
            if speed < 180 and speed > 5:
                _max_speed_sound = 30
                _factor = float(0.15 * speed)
                speeded_array = self.speedx(self.car_sound_array_idle, _factor)
                # continue play
                self.car_sound = pygame.mixer.Sound(speeded_array[min(len(speeded_array), int(self.sound_played_time / 2 / 1e3 * 44100)):])
            self.car_sound.play(maxtime=300)
            self.sound_played_time = 0
        if self._control.throttle == 0:
            self.car_sound.stop()
            self.car_sound_idle.set_volume(0.4)
            self.car_sound_idle.play(maxtime=500)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    @staticmethod
    def speedx(sound_array, factor):
        """ Multiplies the sound's speed by some `factor` """
        indices = np.round(np.arange(0, len(sound_array), factor))
        indices = indices[indices < len(sound_array)].astype(int)
        return sound_array[indices.astype(int)]


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._gauge_display = {}
        self._server_clock = pygame.time.Clock()
        self._passed_waypoints = []
        self.score_from_pengci = 0
        self.score_from_event = 0
        self.deduct_score = 0


        screen_width, screen_height = pygame.display.get_surface().get_size()
        self.pos_x, self.pos_y = int(screen_width * 0.95), int(screen_height * 0.05)


    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        wp = world.map.get_waypoint(t.location)
        self._passed_waypoints.append(wp)
        v = world.player.get_velocity()
        c = world.player.get_control()
        
        self._gauge_display['Speed'] = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        
        self.clock = clock
        


    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        self._notifications.render(display)
        self.help.render(display)


    def render_gauge(self, display):
        gauge_surface = pygame.Surface((300, 300))
        gauge_surface.set_colorkey((0, 0, 0))  # make the background transparent
        gauge_center = (int(gauge_surface.get_width()/2), int(gauge_surface.get_height()/2))
        speed_angle = int(self._gauge_display['Speed'] * 295 / 160)
        speed_color = [(speed_angle % 256), 255 - (speed_angle % 256), 255]  # Cycle through RGB values

        arc_thickness = 50
        _font = pygame.font.SysFont('Arial', 60, bold=True)  

        # Add a glowing effect to the text
        speed_text = _font.render('%15.0f km/h' % self._gauge_display['Speed'], True, speed_color)
        speed_text_back = _font.render('%15.0f km/h' % self._gauge_display['Speed'], True, (0, 0, 0))
        text_rect = speed_text.get_rect(center=(gauge_center[0]-30, gauge_center[1]))
        text_rect.inflate_ip(-3, -3)
        display.blit(speed_text, text_rect.move(3, 3))  # show speed number in neon color
        display.blit(speed_text_back, text_rect)  # show speed number in black on top

        # Create a trailing effect on the progress bar
        for i in range(0, arc_thickness):
            # plot outline
            pygame.gfxdraw.arc(gauge_surface, gauge_center[0], gauge_center[1], 100 - i, -325, 295-325, (55, 77, 91))
            # plot speed with fading effect
            fading_color = [max(0, c - i * 5) for c in speed_color]  # Reduce color intensity as the arc fades out
            pygame.gfxdraw.arc(gauge_surface, gauge_center[0], gauge_center[1], 100 - i, -325, speed_angle-325, fading_color)
        
        display.blit(gauge_surface, (0, 0))  # show speed inde


    
        # # Server FPS
        # server_fps_text = _font.render('Server FPS: %15.0f' % self.server_fps, True, (255, 255, 255)) # White color
        # server_fps_text_rect = server_fps_text.get_rect(topleft=(50, 20))  # Position at top left
        # display.blit(server_fps_text, server_fps_text_rect)

        # # Client FPS
        # client_fps_text = _font.render('Client FPS: %15.0f' % self.clock.get_fps(), True, (255, 255, 255)) # White color
        # client_fps_text_rect = client_fps_text.get_rect(topleft=(50, 50))  # Position below the server FPS
        # display.blit(client_fps_text, client_fps_text_rect)



# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
    
        self.collision_occurred = False  # initialize collide flag
        self.last_collision_frame = 0
        self.collision_frame_threshold = 10
        self.max_intensity = 1000
        self.min_intensity = 20

        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

        if intensity < self.min_intensity:
            self.min_intensity = intensity
        
        if intensity > self.max_intensity:
            self.max_intensity = intensity

        self.history.append((event.frame, intensity))
        
        if len(self.history) > 1000:
            self.history.pop(0)

        # perform score deduction
        if (event.frame - self.last_collision_frame > self.collision_frame_threshold) and (self.collision_occurred == True):
                self.collision_occurred = False

        if not self.collision_occurred:
            # normalize intensity score as a score to deduct
            norm_intensity = (intensity - self.min_intensity) / (self.max_intensity - self.min_intensity)

            # increase deducted scores when collision with pedestrian
            if 'walker' in event.other_actor.type_id:
                norm_intensity = norm_intensity * 50

            self.hud.deduct_score += norm_intensity * 50

            actor_type = get_actor_display_name(event.other_actor)
            self.hud.notification('Collision with %r, deduct  %.2f points!' % (actor_type, norm_intensity * 20))
            
            self.collision_occurred = True
            self.last_collision_frame = event.frame
            


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            # (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
            # (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArm),
            # (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=0, y=0, z=50), carla.Rotation(pitch=-90)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            bp.set_attribute('gamma', '2.2')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        elif self.sensors[self.index][0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

class GameArgs(object):
    def __int__(self):
        self.host = '127.0.0.1'
        self.port = 2000
        self.width, self.height = 1280, 720
        self.scenario = 'OfflineChallenge_1'
        self.autopilot = False
        #self.rolename = 'hero'
        self.keep_ego_vehicle = False



def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.get_world()

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        sim_world.wait_for_tick()

        clock = pygame.time.Clock()


        _event_occur = False
        _event_failed = False
        _event_score = 0
        _event_time = 0.0
        _checkpoint = True

        # game duration in seconds
        game_duration = 10000
        game_start_time = pygame.time.get_ticks() / 1000
        game_remaining_time = max(0, game_duration)
       
        while True:
            clock.tick(60)
            if game_remaining_time <= 0:
                return
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock):
                return
            world.render(display)

            # calculate remaining game time
            game_current_time = pygame.time.get_ticks() / 1000
            game_remaining_time = max(0, game_duration - (game_current_time - game_start_time))

            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            # prevent destruction of ego vehicle
            if args.keep_ego_vehicle:
                world.player = None
            world.destroy()

        pygame.quit()




# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot. This does not autocomplete the scenario')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='role name of ego vehicle to control (default: "hero")')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--keep_ego_vehicle',
        action='store_true',
        help='do not destroy ego vehicle on exit')
    argparser.add_argument(
        '--scenario',
        help='Name of the scenario to be executed. Use the preposition \'group:\' to run all scenarios of one class, e.g. ControlLoss or FollowLeadingVehicle')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
