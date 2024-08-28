import math
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class RobotParticleEnv(gym.Env):
    def __init__(self, env_config=None):
        super(RobotParticleEnv, self).__init__()
        self.render_mode = env_config.get("render_mode")

        # Rendering settings
        self.FPS = 24
        self.dt = 1.0 / self.FPS
        self.viewer = None

        # Environment settings
        self.max_episode_time = 150

        self.command_time_interval = None
        self.min_command_time_interval = 1  # self.dt
        self.max_command_time_interval = 1

        self.max_forward_speed = None
        self.min_max_forward_speed = 20.0
        self.max_max_forward_speed = 20.0

        self.max_backward_speed = None
        self.min_max_backward_speed = 10.0
        self.max_max_backward_speed = 10.0

        # Robots and particles settings
        self.min_robots = 1
        self.max_robots = 1
        self.min_robot_radius = 10
        self.max_robot_radius = 20
        self.robot_capture_angle = 45

        self.num_particles = None
        self.min_particles = 1
        self.max_particles = 1
        self.min_particle_radius = 10
        self.max_particle_radius = 30

        # Simulation settings
        self.world_width = 600
        self.world_height = 500
        self.state = None
        self.last_state = None
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                self.max_robots * 2,
            ),  # 2 actions per robot (lfeftSpeed, rightSpeed)
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                # 8 * robots: (position (x, y), radius, angle, leftSpeed,
                #              rightSpeed, colliding, active)
                self.max_robots * 8
                # 2 * robots * particles: (distance (x, y))
                + self.max_robots * self.max_particles * 2
                # 5: particles (position (x, y), radius, dead, active)
                + self.max_particles * 5
                # 3: command_time_interval, max_forward_speed, max_backward_speed
                + 3,
            ),
            dtype=np.float32,
        )

    def step(self, actions):
        for i, robot in enumerate(self.state["robots"]):
            left_speed = actions[i * 2] * 2 - 1  # [0, 1] -> [-1, 1]
            robot["leftSpeed"] = (
                left_speed * self.max_forward_speed
                if left_speed >= 0
                else left_speed * self.max_backward_speed
            )
            right_speed = actions[i * 2 + 1] * 2 - 1  # [0, 1] -> [-1, 1]
            robot["rightSpeed"] = (
                right_speed * self.max_forward_speed
                if right_speed >= 0
                else right_speed * self.max_backward_speed
            )

            if math.isclose(robot["leftSpeed"], robot["rightSpeed"], rel_tol=1e-4):
                robot["leftSpeed"] = robot["rightSpeed"]
            elif math.isclose(robot["leftSpeed"], -robot["rightSpeed"], rel_tol=1e-4):
                robot["leftSpeed"] = -robot["rightSpeed"]
            elif math.isclose(robot["rightSpeed"], -robot["leftSpeed"], rel_tol=1e-4):
                robot["rightSpeed"] = -robot["leftSpeed"]

        self._update()

        # Observations
        is_time_finished = self.state["time"] >= self.max_episode_time
        all_particles_dead = all(
            [p["dead"] for p in self.state["particles"] if p["active"]]
        )

        observations = self._get_observations()
        assert not np.any(np.isnan(observations))
        truncated = is_time_finished

        # Rewards
        reward = -0.1
        if self.done:
            reward = -1.0
        elif all_particles_dead:
            reward = 1.0
            self.done = True
        elif self.last_state:
            if self.last_state["total_score"] < self.state["total_score"]:
                reward = 0.5
            # Reward robots positively for moving towards particles
            else:
                last_dist = min(
                    [
                        r["min_dist_to_particle"]
                        for r in self.last_state["robots"]
                        if r["active"]
                    ]
                )
                dist = min(
                    [
                        r["min_dist_to_particle"]
                        for r in self.state["robots"]
                        if r["active"]
                    ]
                )
                diff = (last_dist - dist) / max(self.world_width, self.world_height)
                reward = diff
                # reward = -(dist / max(self.world_width, self.world_height)) / 10

        self.last_state = deepcopy(self.state)

        return observations, reward, self.done, truncated, {}  # info

    def reset(self, *, seed=None, options=None):
        # Reset the environment to its initial state
        self.num_particles = np.random.randint(
            self.min_particles, self.max_particles + 1
        )
        self.state = self._get_initial_state()
        self.last_state = None
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
        return self._get_observations(), {}

    def render(self, mode="human"):
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode((self.world_width, self.world_height))
            pygame.display.set_caption("VroumBotRL")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self.viewer.fill((200, 200, 200))  # Gray background

        # Draw robots
        for robot in self.state["robots"]:
            if not robot["active"]:
                continue
            pygame.draw.circle(
                self.viewer,
                (255, 255, 255),
                (int(robot["position"]["x"]), int(robot["position"]["y"])),
                int(robot["radius"]),
            )
            pygame.draw.circle(
                self.viewer,
                (0, 0, 0),
                (int(robot["position"]["x"]), int(robot["position"]["y"])),
                int(robot["radius"]),
                1,
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
            if not particle["active"] or particle["dead"]:
                continue
            pygame.draw.circle(
                self.viewer,
                (255, 150, 150),
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
                    "captureAngle": self.robot_capture_angle,
                    "id": i,
                    "leftSpeed": np.random.uniform(
                        -self.max_max_backward_speed, self.max_max_forward_speed
                    ),
                    "rightSpeed": np.random.uniform(
                        -self.max_max_backward_speed, self.max_max_forward_speed
                    ),
                    "position": {
                        "x": x,
                        "y": y,
                    },
                    "radius": radius,
                    "score": 0.0,
                    "colliding": False,
                    "min_dist_to_particle": max(self.world_width, self.world_height),
                    "active": 1 if i < num_robots else 0,
                }
            )

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
                    "explosionTimes": [[self.max_episode_time + 1]],
                    "dead": False,
                    "active": 1 if i < self.num_particles else 0,
                }
            )

        return {
            "robots": robots,
            "particles": particles,
            "time": 0.0,
            "worldEnd": {"x": self.world_width, "y": self.world_height},
            "worldOrigin": {"x": 0, "y": 0},
            # "num_active_robots": num_robots,
            # "num_active_particles": self.num_particles,
            "total_score": 0.0,
        }

    def _normalize(self, value: float, low: float, high: float):
        # check if low values are same to high to avoid division by zero
        if low == high:
            return 1.0
        return (value - low) / (high - low)

    def _one_hot(self, value, num_classes):
        return np.eye(num_classes)[value]

    def _get_observations(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for i, robot in enumerate(self.state["robots"]):
            offset = i * 8
            obs[offset] = self._normalize(robot["position"]["x"], 0, self.world_width)
            obs[offset + 1] = self._normalize(
                robot["position"]["y"], 0, self.world_height
            )
            obs[offset + 2] = self._normalize(
                robot["radius"], self.min_robot_radius, self.max_robot_radius
            )
            obs[offset + 3] = self._normalize(robot["angle"], 0, 360)
            obs[offset + 4] = self._normalize(
                robot["leftSpeed"], -self.max_backward_speed, self.max_forward_speed
            )
            obs[offset + 5] = self._normalize(
                robot["rightSpeed"], -self.max_backward_speed, self.max_forward_speed
            )
            obs[offset + 6] = robot["colliding"]
            obs[offset + 7] = robot["active"]
            for i, particle in enumerate(self.state["particles"]):
                if not particle["active"] or particle["dead"]:
                    continue
                offset = self.max_robots * 8 + i * 2
                obs[offset] = self._normalize(
                    abs(particle["position"]["x"] - robot["position"]["x"]),
                    0,
                    self.world_width,
                )
                obs[offset + 1] = self._normalize(
                    abs(particle["position"]["y"] - robot["position"]["y"]),
                    0,
                    self.world_height,
                )

        for i, particle in enumerate(self.state["particles"]):
            offset = self.max_robots * 8 + self.max_particles * 2 + i * 5
            obs[offset + 3] = particle["dead"]
            obs[offset + 4] = particle["active"]
            if not particle["active"] or particle["dead"]:
                continue
            obs[offset] = self._normalize(
                particle["position"]["x"], 0, self.world_width
            )
            obs[offset + 1] = self._normalize(
                particle["position"]["y"], 0, self.world_height
            )
            obs[offset + 2] = self._normalize(
                particle["radius"], self.min_particle_radius, self.max_particle_radius
            )

        offset = self.max_robots * 8 + self.max_particles * 2 + self.max_particles * 5
        obs[offset] = self._normalize(
            self.command_time_interval,
            self.min_command_time_interval,
            self.max_command_time_interval,
        )
        obs[offset + 1] = self._normalize(
            self.max_forward_speed,
            self.min_max_forward_speed,
            self.max_max_forward_speed,
        )
        obs[offset + 2] = self._normalize(
            self.max_backward_speed,
            self.min_max_backward_speed,
            self.max_max_backward_speed,
        )
        return obs

    def _update(self):
        elapsed_time = 0
        while elapsed_time < self.command_time_interval:
            self._check_collisions()
            self._update_robots()
            # self._update_particles()
            self.state["time"] += self.dt
            elapsed_time += self.dt
            if self.render_mode == "human" and elapsed_time % 0.1 < self.dt:
                self.render()

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
            robot["angle"] = robot["angle"] % 360
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
            if not particle["active"] or particle["dead"]:
                continue
            if particle["explosionTimes"][0][0] <= self.state["time"]:
                self._particle_explosion(i)

    def _particle_explosion(self, i):
        if len(self.state["particles"][i]["explosionTimes"]) == 1:
            self.state["particles"][i]["dead"] = True
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
            robot["colliding"] = False
            if not robot["active"] or self.done:
                continue
            robot["min_dist_to_particle"] = max(self.world_width, self.world_height)
            for particle in self.state["particles"]:
                if not particle["active"] or particle["dead"] or self.done:
                    continue
                robot["min_dist_to_particle"] = min(
                    robot["min_dist_to_particle"],
                    self._distance_to_object(robot, particle),
                )
                if self._is_collision(robot, particle):
                    # Check if the robot is capturing the particle
                    # Robots has angle in degrees
                    angle_robot_particle = (
                        math.degrees(
                            math.atan2(
                                particle["position"]["y"] - robot["position"]["y"],
                                particle["position"]["x"] - robot["position"]["x"],
                            )
                        )
                        + 360
                    ) % 360
                    angle_diff = abs(angle_robot_particle - robot["angle"])
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    if angle_diff <= robot["captureAngle"]:
                        particle["dead"] = True
                        score = math.pi * particle["radius"] ** 2
                        self.state["robots"][i]["score"] += score
                        self.state["total_score"] += score
                    else:
                        # self.done = True
                        robot["colliding"] = True
                        robot_next = deepcopy(robot)
                        self._update_robot(robot_next)
                        self._update_robot(robot_next)
                        if self._is_collision(robot_next, particle):
                            if robot["leftSpeed"] + robot["rightSpeed"] != 0:
                                self.state["robots"][i]["leftSpeed"] = 0
                                self.state["robots"][i]["rightSpeed"] = 0
            for k in range(i + 1, len(self.state["robots"])):
                other_robot = self.state["robots"][k]
                if not other_robot["active"] or self.done:
                    continue
                if self._is_collision(robot, other_robot):
                    self.done = True
                    robot["colliding"] = True
                    other_robot["colliding"] = True
                    # robot_next = deepcopy(robot)
                    # other_robot_next = deepcopy(other_robot)
                    # self._update_robot(robot_next)
                    # self._update_robot(robot_next)
                    # self._update_robot(other_robot_next)
                    # self._update_robot(other_robot_next)
                    # if self._is_collision(robot_next, other_robot_next):
                    #     if robot["leftSpeed"] + robot["rightSpeed"] != 0:
                    #         robot["leftSpeed"] = 0
                    #         robot["rightSpeed"] = 0
                    #     if other_robot["leftSpeed"] + other_robot["rightSpeed"] != 0:
                    #         other_robot["leftSpeed"] = 0
                    #         other_robot["rightSpeed"] = 0

    def _is_collision(self, object1, object2):
        if (
            self._distance_to_object(object1, object2)
            <= object1["radius"] + object2["radius"]
        ):
            return True
        return False

    def _distance_to_object(self, object1, object2):
        return math.sqrt(
            pow(object1["position"]["x"] - object2["position"]["x"], 2)
            + pow(object1["position"]["y"] - object2["position"]["y"], 2)
        )

    def clamp(self, n, smallest, largest):
        return max(smallest, min(n, largest))
