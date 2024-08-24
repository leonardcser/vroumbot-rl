import math
from copy import deepcopy

import gym
import numpy as np
import pygame
from gym import spaces


class RobotParticleEnv(gym.Env):
    def __init__(self):
        super(RobotParticleEnv, self).__init__()

        # Rendering settings
        self.FPS = 24
        self.dt = 1.0 / self.FPS
        self.viewer = None

        # Environment settings
        self.max_episode_time = 100

        self.command_time_interval = 1
        self.min_command_time_interval = 1  # self.dt
        self.max_command_time_interval = 1

        self.max_forward_speed = 20.0
        self.min_max_forward_speed = 20.0
        self.max_max_forward_speed = 20.0

        self.max_backward_speed = 5.0
        self.min_max_backward_speed = 5.0
        self.max_max_backward_speed = 5.0

        # Robots and particles settings
        self.min_robots = 1
        self.max_robots = 1
        self.min_robot_radius = 10
        self.max_robot_radius = 20

        self.min_particles = 5
        self.max_particles = 10
        self.min_particle_radius = 10
        self.max_particle_radius = 30

        # Simulation settings
        self.world_width = 600
        self.world_height = 500
        self.state = None
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Box(
            low=-self.max_backward_speed,
            high=self.max_forward_speed,
            shape=(self.max_robots * 2,),  # 2 actions per robot (left and right wheel)
            dtype=np.float32,
        )

        # Observation space for each robot
        robot_obs_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=0,
                    high=max(self.world_width, self.world_height),
                    shape=(2,),
                    dtype=np.float32,
                ),
                "radius": spaces.Box(
                    low=self.min_robot_radius,
                    high=self.max_robot_radius,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "angle": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32),
                "leftSpeed": spaces.Box(
                    low=-self.max_backward_speed,
                    high=self.max_forward_speed,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "rightSpeed": spaces.Box(
                    low=-self.max_backward_speed,
                    high=self.max_forward_speed,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "active": spaces.Discrete(2),  # 0 for inactive, 1 for active
            }
        )

        particle_obs_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=self.min_particle_radius,
                    high=max(self.world_width, self.world_height)
                    - self.min_particle_radius,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "radius": spaces.Box(
                    low=self.min_particle_radius,
                    high=self.max_particle_radius,
                    shape=(1,),
                    dtype=np.float32,
                ),
                # "explosionTimes": spaces.Sequence(
                #     spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
                # ),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "robots": spaces.Tuple((robot_obs_space,) * self.max_robots),
                "particles": spaces.Tuple((particle_obs_space,) * self.max_particles),
                "command_time_interval": spaces.Box(
                    low=self.min_command_time_interval,
                    high=self.max_command_time_interval,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "max_forward_speed": spaces.Box(
                    low=self.min_max_forward_speed,
                    high=self.max_max_forward_speed,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "max_backward_speed": spaces.Box(
                    low=self.min_max_backward_speed,
                    high=self.max_max_backward_speed,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "num_active_robots": spaces.Discrete(self.max_robots + 1),
                "num_active_particles": spaces.Discrete(self.max_particles + 1),
            }
        )

    def step(self, actions):
        for i, robot in enumerate(self.state["robots"]):
            if robot["active"]:
                robot["leftSpeed"] = actions[i * 2]
                robot["rightSpeed"] = actions[i * 2 + 1]
                if math.isclose(robot["leftSpeed"], robot["rightSpeed"], rel_tol=1e-5):
                    robot["leftSpeed"] = robot["rightSpeed"]

        self._update()

        if self.state["time"] >= self.max_episode_time:
            self.done = True

        observations = self._get_observations()
        assert observations.keys() == self.observation_space.spaces.keys()
        if not self.done:
            rewards = [
                robot["score"] for robot in self.state["robots"] if robot["active"]
            ]
        else:
            rewards = [-100.0] * self.max_robots
        info = {}

        return observations, rewards, self.done, info

    def reset(self):
        # Reset the environment to its initial state
        self.state = self._get_initial_state()
        self.command_time_interval = np.random.uniform(
            self.min_command_time_interval, self.max_command_time_interval
        )
        self.max_forward_speed = np.random.uniform(
            self.min_max_forward_speed, self.max_max_forward_speed
        )
        self.max_backward_speed = np.random.uniform(
            self.min_max_backward_speed, self.max_max_backward_speed
        )
        self.done = False
        return self._get_observations()

    def render(self, mode="human"):
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((self.world_width, self.world_height))
            pygame.display.set_caption("VroumBotRL")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self.viewer.fill((255, 255, 255))  # White background

        # Draw robots
        for robot in self.state["robots"]:
            if not robot["active"]:
                continue
            pygame.draw.circle(
                self.viewer,
                (0, 0, 255),
                (int(robot["position"]["x"]), int(robot["position"]["y"])),
                int(robot["radius"]),
            )
            # Draw robot "mouth" which indicates the capture angle
            mouth_angle_1 = (robot["angle"] + robot["captureAngle"] / 2) % 360
            mouth_angle_2 = (robot["angle"] - robot["captureAngle"] / 2) % 360
            # Draw triangle for the robot's capture angle
            pygame.draw.polygon(
                self.viewer,
                (0, 0, 0),
                [
                    (
                        robot["position"]["x"],
                        robot["position"]["y"],
                    ),
                    (
                        robot["position"]["x"]
                        + robot["radius"] * math.cos(math.radians(mouth_angle_1)),
                        robot["position"]["y"]
                        + robot["radius"] * math.sin(math.radians(mouth_angle_1)),
                    ),
                    (
                        robot["position"]["x"]
                        + robot["radius"] * math.cos(math.radians(mouth_angle_2)),
                        robot["position"]["y"]
                        + robot["radius"] * math.sin(math.radians(mouth_angle_2)),
                    ),
                ],
            )

        # Draw particles
        for particle in self.state["particles"]:
            if not particle["active"]:
                continue
            pygame.draw.circle(
                self.viewer,
                (0, 255, 0),
                (int(particle["position"]["x"]), int(particle["position"]["y"])),
                int(particle["radius"]),
            )
            # draw particle id in center of particle
            font = pygame.font.Font(None, int(particle["radius"]))
            text = font.render(str(particle["id"]), True, (0, 0, 0))
            text_rect = text.get_rect(
                center=(int(particle["position"]["x"]), int(particle["position"]["y"]))
            )
            self.viewer.blit(text, text_rect)

        pygame.display.flip()

    def close(self):
        if self.viewer:
            pygame.quit()
            self.viewer = None

    def _get_initial_state(self):
        num_robots = np.random.randint(self.min_robots, self.max_robots + 1)
        robots = []
        for i in range(self.max_robots):
            radius = np.random.uniform(self.min_robot_radius, self.max_robot_radius)
            x = np.random.uniform(radius, self.world_width - radius)
            y = np.random.uniform(radius, self.world_height - radius)
            found = False
            while len(robots) > 0 and not found:
                for robot in robots:
                    if self._is_collision(
                        robot, {"position": {"x": x, "y": y}, "radius": radius}
                    ):
                        x = np.random.uniform(radius, self.world_width - radius)
                        y = np.random.uniform(radius, self.world_height - radius)
                        break
                else:
                    found = True

            robots.append(
                {
                    "angle": np.random.uniform(0, 360),
                    # TODO: Randomize capture angle
                    "captureAngle": 45.0,
                    "id": i,
                    "leftSpeed": np.random.uniform(
                        -self.max_backward_speed, self.max_forward_speed
                    ),
                    "rightSpeed": np.random.uniform(
                        -self.max_backward_speed, self.max_forward_speed
                    ),
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "radius": radius,
                    "score": 0.0,
                    "active": 1 if i < num_robots else 0,
                }
            )

        num_particles = np.random.randint(self.min_particles, self.max_particles + 1)
        particles = []
        for i in range(self.max_particles):
            radius = np.random.uniform(
                self.min_particle_radius, self.max_particle_radius
            )
            x = np.random.uniform(radius, self.world_width - radius)
            y = np.random.uniform(radius, self.world_height - radius)
            found = False
            while not found:
                for object in particles + robots:
                    if self._is_collision(
                        object, {"position": {"x": x, "y": y}, "radius": radius}
                    ):
                        x = np.random.uniform(radius, self.world_width - radius)
                        y = np.random.uniform(radius, self.world_height - radius)
                        break
                else:
                    found = True
            particles.append(
                {
                    "id": i,
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "radius": radius,
                    "explosionTimes": [[self.max_episode_time]],
                    "active": 1 if i < num_particles else 0,
                }
            )

        return {
            "robots": robots,
            "particles": particles,
            "time": 0.0,
            "worldEnd": {"x": self.world_width, "y": self.world_height},
            "worldOrigin": {"x": 0, "y": 0},
            "num_active_robots": num_robots,
            "num_active_particles": num_particles,
        }

    def _normalize(self, value, obs_space):
        low = obs_space.low
        high = obs_space.high
        # check if low values are same to high to avoid division by zero
        if (low == high).all():
            return 1.0
        return (value - low) / (high - low)

    def _get_observations(self):
        obs_space = self.observation_space
        robot_obs_space = obs_space["robots"][0].spaces
        particle_obs_space = obs_space["particles"][0].spaces
        return {
            "robots": [
                {
                    "position": np.array(
                        [
                            self._normalize(
                                robot["position"]["x"], robot_obs_space["position"]
                            ),
                            self._normalize(
                                robot["position"]["y"], robot_obs_space["position"]
                            ),
                        ],
                        dtype=np.float32,
                    ),
                    "radius": np.array(
                        [self._normalize(robot["radius"], robot_obs_space["radius"])],
                        dtype=np.float32,
                    ),
                    "angle": np.array(
                        [self._normalize(robot["angle"], robot_obs_space["angle"])],
                        dtype=np.float32,
                    ),
                    "leftSpeed": np.array(
                        [
                            self._normalize(
                                robot["leftSpeed"], robot_obs_space["leftSpeed"]
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "rightSpeed": np.array(
                        [
                            self._normalize(
                                robot["rightSpeed"], robot_obs_space["rightSpeed"]
                            )
                        ],
                        dtype=np.float32,
                    ),
                    "active": robot["active"],
                }
                for robot in self.state["robots"]
            ],
            "particles": [
                {
                    "position": np.array(
                        [
                            self._normalize(
                                particle["position"]["x"],
                                particle_obs_space["position"],
                            ),
                            self._normalize(
                                particle["position"]["y"],
                                particle_obs_space["position"],
                            ),
                        ],
                        dtype=np.float32,
                    ),
                    "radius": np.array(
                        [
                            self._normalize(
                                particle["radius"], particle_obs_space["radius"]
                            )
                        ],
                        dtype=np.float32,
                    ),
                    # "explosionTimes": particle["explosionTimes"],
                }
                for particle in self.state["particles"]
            ],
            "command_time_interval": np.array(
                [
                    self._normalize(
                        self.command_time_interval, obs_space["command_time_interval"]
                    )
                ],
                dtype=np.float32,
            ),
            "max_forward_speed": np.array(
                [
                    self._normalize(
                        self.max_forward_speed, obs_space["max_forward_speed"]
                    )
                ],
                dtype=np.float32,
            ),
            "max_backward_speed": np.array(
                [
                    self._normalize(
                        self.max_backward_speed, obs_space["max_backward_speed"]
                    )
                ],
                dtype=np.float32,
            ),
            "num_active_robots": self.state["num_active_robots"],
            "num_active_particles": self.state["num_active_particles"],
        }

    def _update(self):
        elapsed_time = 0
        while elapsed_time < self.command_time_interval:
            self._check_collisions()
            self._update_robots()
            # self._update_particles()
            self.state["time"] += self.dt
            elapsed_time += self.dt

    def _update_robots(self):
        for robot in self.state["robots"]:
            if not robot["active"]:
                continue
            # Check if the robot is out of bounds
            if (
                robot["position"]["x"] - robot["radius"] < 0
                or robot["position"]["x"] + robot["radius"] > self.world_width
                or robot["position"]["y"] - robot["radius"] < 0
                or robot["position"]["y"] + robot["radius"] > self.world_height
            ):
                self.done = True
                break
            self._update_robot(robot)

    def _update_robot(self, robot):
        radius = robot["radius"]
        angle_rad = math.radians(robot["angle"])
        leftSpeed = robot["leftSpeed"]
        rightSpeed = robot["rightSpeed"]
        vt = (leftSpeed + rightSpeed) / 2
        if leftSpeed == rightSpeed:
            # Move the robot straight
            robot["position"]["x"] += vt * self.dt * math.cos(angle_rad)
            robot["position"]["y"] += vt * self.dt * math.sin(angle_rad)
        elif leftSpeed != -rightSpeed:
            R = radius * (leftSpeed + rightSpeed) / (leftSpeed - rightSpeed)
            omega = vt / R
            # Update robot positions based on their speeds
            robot["position"]["x"] += R * (
                math.sin(omega * self.dt + angle_rad) - math.sin(angle_rad)
            )
            robot["position"]["y"] -= R * (
                math.cos(omega * self.dt + angle_rad) - math.cos(angle_rad)
            )
            # Update robot angles based on their speeds
            robot["angle"] = (robot["angle"] + math.degrees(omega * self.dt)) % 360
        else:
            # Robot is turning on the spot
            robot["angle"] = (robot["angle"] + leftSpeed * self.dt) % 360
        assert not math.isnan(robot["position"]["x"])
        assert not math.isnan(robot["position"]["y"])

    def _update_particles(self):
        for i, particle in enumerate(self.state["particles"]):
            if not particle["active"]:
                continue
            if particle["explosionTimes"][0][0] <= self.state["time"]:
                self._particle_explosion(i)

    def _particle_explosion(self, i):
        if len(self.state["particles"][i]["explosionTimes"]) == 1:
            self.state["particles"].pop(i)
            return

        new_r = self.state["particles"][i]["radius"] / (1 + math.sqrt(2))
        new_id = self.state["particles"][-1]["id"]
        x = self.state["particles"][i]["position"]["x"]
        y = self.state["particles"][i]["position"]["y"]

        for j in range(1, 5):
            child_particule = {
                "id": new_id + j,
                "radius": new_r,
                "position": {"x": 0, "y": 0},
                "explosionTimes": [],
            }

            if j <= 2:
                child_particule["position"]["x"] = x - new_r
            else:
                child_particule["position"]["x"] = x + new_r

            if j % 2:
                child_particule["position"]["y"] = y - new_r
            else:
                child_particule["position"]["y"] = y + new_r

            for k in range(1, len(self.state["particles"][i]["explosionTimes"])):
                times = []
                for l in range((j - 1) * pow(4, k - 1), pow(4, k - 1) * j):
                    times.append(self.state["particles"][i]["explosionTimes"][k][l])
                child_particule["explosionTimes"].append(times)

            self.state["particles"].append(child_particule)

        self.state["particles"].pop(i)

    def _check_collisions(self):
        for i, robot in enumerate(self.state["robots"]):
            if not robot["active"]:
                continue
            for j, particle in enumerate(self.state["particles"]):
                if not particle["active"]:
                    continue
                if self._is_collision(robot, particle):
                    # Check if capture angle is in the direction of the particle
                    if (
                        abs(
                            math.atan2(
                                particle["position"]["y"] - robot["position"]["y"],
                                particle["position"]["x"] - robot["position"]["x"],
                            )
                            - math.radians(robot["angle"])
                        )
                        < math.radians(robot["captureAngle"]) / 2
                    ):
                        particule = self.state["particles"].pop(j)
                        self.state["robots"][i]["score"] += particule["radius"]
                    else:
                        robot_next = deepcopy(robot)
                        self._update_robot(robot_next)
                        if self._is_collision(robot_next, particle):
                            if robot["leftSpeed"] + robot["rightSpeed"] != 0:
                                self.state["robots"][i]["leftSpeed"] = 0
                                self.state["robots"][i]["rightSpeed"] = 0
            for k in range(i + 1, len(self.state["robots"])):
                other_robot = self.state["robots"][k]
                if not other_robot["active"]:
                    continue
                if self._is_collision(robot, other_robot):
                    robot_next = deepcopy(robot)
                    other_robot_next = deepcopy(other_robot)
                    self._update_robot(robot_next)
                    self._update_robot(other_robot_next)
                    if self._is_collision(robot_next, other_robot_next):
                        if robot["leftSpeed"] + robot["rightSpeed"] != 0:
                            robot["leftSpeed"] = 0
                            robot["rightSpeed"] = 0
                        if other_robot["leftSpeed"] + other_robot["rightSpeed"] != 0:
                            other_robot["leftSpeed"] = 0
                            other_robot["rightSpeed"] = 0

    def _is_collision(self, cicle1, circle2):
        if (
            math.sqrt(
                pow(cicle1["position"]["x"] - circle2["position"]["x"], 2)
                + pow(cicle1["position"]["y"] - circle2["position"]["y"], 2)
            )
            <= cicle1["radius"] + circle2["radius"]
        ):
            return True
        return False
