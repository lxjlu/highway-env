from highway_env.envs import Action
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
                # "type": "ContinuousAction",
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "policy_frequency": 10,
            'simulation_frequency': 10,
            "vehicles_count": 1,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "centering_position": [0.5, 0.5],
            "screen_width": 789,
            "screen_height": 400,
            "duration": 100,
            "reward_speed_range": [8, 24],
            "offroad_terminal": False,
            "show_trajectories": True,
            "scaling": 9,

            "v_lane_id": ("a", "b", 1),
            "v_target_id": ("a", "b", 1),
            "v_x": 100.0,
            "v_y": 0.0,
            "v_h": 0.0,
            "v_s": 10.0,
            "v_target_s": 20.0,
            "KP_A": 2,
            "KP_HEADING": 2,
            "KP_LATERAL": 3,
            "TAU_PURSUIT": 0.1,

            # "radius": np.random.choice([25, 50, 100, 250, 500, 1000]),
            "radius": np.random.choice([250]),
        })
        return config

    def _make_road(self) -> None:
        net = RoadNetwork()

        radius = self.config["radius"]  # [m]
        # radius = np.random.choice([50, 500])  # [m]
        # radius = np.random.choice([50, 200, 500, 1000])  # [m]
        center = [0, StraightLane.DEFAULT_WIDTH + radius]  # [m]
        alpha = 0  # [deg]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH,
                 radius + 2 * StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, s], [n, c]]

        for lane in [0, 1, 2]:
            net.add_lane("a", "b",
                         # CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-90+alpha),
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(-360 + alpha),
                                      clockwise=False, line_types=line[lane]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road
        self.road.record_history = True

    def _make_vehicles(self):
        ego_lane = self.road.network.get_lane(self.config["v_lane_id"])
        ego_vehicle = LxVehicle(self.road,
                                ego_lane.position(self.config["v_x"], self.config["v_y"]),
                                heading=self.config["v_h"],
                                speed=self.config["v_s"],
                                target_lane_index=self.config["v_target_id"],
                                target_speed=self.config["v_target_s"]
                                )

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _reward(self, action):
        pass

    def _is_terminal(self) -> bool:
        pass

    def _cost(self, action: Action) -> float:
        pass


register(
    id='myenv-r1-v0',
    entry_point='highway_env.envs:MyEnv'
)
