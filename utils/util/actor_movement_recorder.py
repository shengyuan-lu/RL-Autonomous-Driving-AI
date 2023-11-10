import csv

class MovementRecorder:
    def __init__(self, actor):
        self._movements_data = []
        self._actor = actor

    def record_movement(self):
        location = self._actor.get_location()
        rotation = self._actor.get_transform().rotation
        self._movements_data.append((location.x, location.y, location.z, rotation.yaw))

    def get_movements_data(self):
        return self._movements_data
