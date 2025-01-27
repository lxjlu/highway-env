from typing import List, Tuple, Union

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle
from highway_env.utils import lmap


class LxVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    # TAU_ACC = 0.6  # [s]
    # TAU_HEADING = 0.2  # [s]
    # TAU_LATERAL = 0.6  # [s]

    # TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    # KP_A = 1 / TAU_ACC
    # KP_HEADING = 1 / TAU_HEADING
    # KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None,
                 KP_A=2,
                 KP_HEADING=2,
                 KP_LATERAL=3,
                 TAU_PURSUIT=0.1,
                 ):
        super().__init__(road, position, heading, speed)
        self.TAU_PURSUIT = TAU_PURSUIT
        self.KP_LATERAL = KP_LATERAL
        self.KP_A = KP_A
        self.KP_HEADING = KP_HEADING
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "LxVehicle") -> "LxVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination: str) -> "LxVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def target_lane_position(self, initial_position, road_r):
        target_lane = self.road.network.get_lane(self.target_lane_index)
        lane_coords = target_lane.local_coordinates(initial_position)
        # x_r = lane_coords[0] + np.arange(0, 301, 10)
        # lane_x = [target_lane.position(item, 0)[0] for item in x_r]
        # lane_y = [0 for item in x_r]
        x_r = lane_coords[0]
        y_r = 0
        lane_x = []
        lane_x_n = []
        lane_y = []
        lane_y_n = []
        for item in np.arange(0, 101, 10):
            xx, yy = target_lane.position(x_r + item, 0)
            xx_n = lmap(xx, [-(road_r+4), (road_r+4), 254], [-1, 1])
            yy_n = lmap(yy, [0, 2*(road_r+4)], [-1, 1])
            lane_x.append(xx)
            lane_x_n.append(xx_n)
            lane_y.append(yy)
            lane_y_n.append(yy_n)
        return lane_x, lane_y, lane_x_n, lane_y_n

    def target_lane2_position(self):
        target_lane = self.road.network.get_lane(self.target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        x, y = target_lane.position(lane_coords[0], 0)
        return x, y

    def act(self, action: Union[dict, str] = None, PID=True) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        # if action == "FASTER":
        #     self.target_speed += self.DELTA_SPEED
        # elif action == "SLOWER":
        #     self.target_speed -= self.DELTA_SPEED
        # elif action == "LANE_RIGHT":
        #     _from, _to, _id = self.target_lane_index
        #     target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        #     if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
        #         self.target_lane_index = target_lane_index
        # elif action == "LANE_LEFT":
        #     _from, _to, _id = self.target_lane_index
        #     target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        #     if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
        #         self.target_lane_index = target_lane_index
        if PID:
            action = {"steering": self.steering_control(self.target_lane_index),
                      "acceleration": self.speed_control(self.target_speed)}
            action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """

        return np.clip(self.KP_A * (target_speed - self.speed), -5.0, 5.0)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index + 1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                           for t in times]))
