from highway_env.vehicle.controller import MDPVehicle, ControlledVehicle

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import RoadNetwork, Road

import numpy as np
from typing import Tuple, Optional
from gym.envs.registration import register

from highway_env.vehicle.lx_vehicle import LxVehicle


class MyEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "policy_frequency": 10,
            'simulation_frequency': 10,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "centering_position": [0.6, 0.8],
            "screen_width": 789,
            "screen_height": 400,
            "duration": 100,
            "reward_speed_range": [8, 24],
            "offroad_terminal": False
        })
        return config

    def _make_road(self):
        net = RoadNetwork()

        radius = 200  # [m]
        center = [10, StraightLane.DEFAULT_WIDTH + radius]  # [m]
        alpha = 0  # [deg]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH,
                 radius + 2 * StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, s], [n, c]]

        for lane in [0, 1, 2]:
            net.add_lane("a", "b",
                         # CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90+alpha),
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        ego_lane = self.road.network.get_lane(("a", "b", 1))
        ego_vehicle = LxVehicle(self.road,
                                ego_lane.position(20, 0),
                                target_lane_index=("a", "b", 0),
                                speed=20)

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _reward(self, action):
        pass

    def _is_terminal(self) -> bool:
        pass


register(
    id='myenv-r1-v0',
    entry_point='highway_env.envs:MyEnv'
)
