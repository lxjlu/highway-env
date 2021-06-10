from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import RoadNetwork, Road

import numpy as np
from typing import Tuple, Optional
from gym.envs.registration import register


class MyEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "ContinuousAction",
            },
            "centering_position": [0.1, 0.8],
            "screen_width": 789,
            "screen_height": 400,
            "duration": 10,
            "reward_speed_range": [8, 24],
            "offroad_terminal": False
        })
        return config

    def _make_road(self):
        net = RoadNetwork()

        radius = 20  # [m]
        center = [10, StraightLane.DEFAULT_WIDTH + radius]  # [m]
        alpha = 0  # [deg]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH,
                 radius + 2 * StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, s], [n, c]]

        for lane in [0, 1, 2]:
            net.add_lane("a", "b",
                         # CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90+alpha),
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-0 + alpha),
                                      clockwise=False, line_types=line[lane]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        ego_lane = self.road.network.get_lane(("a", "b", 1))
        ego_vehicle = self.action_type.vehicle_class(self.road, \
                                                     ego_lane.position(0, 0),
                                                     speed=16)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def define_spaces(self) -> None:
        return super().define_spaces()

    def step(self, action):
        pass

    def _reward(self, action):
        pass

    def _is_terminal(self) -> bool:
        pass


register(
    id='myenv-r1-v0',
    entry_point='highway_env.envs:MyEnv'
)
