import math
from copy import deepcopy

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class RobotParticleEnv(gym.Env):
    def __init__(self, env_config=None):
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

        self.max_backward_speed = 10.0
        self.min_max_backward_speed = 10.0
        self.max_max_backward_speed = 10.0

        # Robots and particles settings
        self.min_robots = 1
        self.max_robots = 1
        self.min_robot_radius = 10
        self.max_robot_radius = 20
        self.robot_capture_angle = 45

        self.min_particles = 5
        self.max_particles = 5
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

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                # 8 * robots: (position (x, y), radius, angle, leftSpeed,
                #              rightSpeed, colliding, active)
                self.max_robots * 8
                # 4: particles (position (x, y), radius, active)
                + self.max_particles * 4
                # 3: command_time_interval, max_forward_speed, max_backward_speed
                + 3
                # 1 * robots: num_active_robots
                + (self.max_robots + 1)
                # 1 * particules: num_active_particles
                + (self.max_particles + 1),
            ),
            dtype=np.float32,
        )

    def step(self, actions):
        for i, robot in enumerate(self.state["robots"]):
            if robot["active"]:
                robot["leftSpeed"] = actions[i * 2]
                robot["rightSpeed"] = actions[i * 2 + 1]
                if math.isclose(robot["leftSpeed"], robot["rightSpeed"], rel_tol=1e-5):
                    robot["leftSpeed"] = robot["rightSpeed"]

        particles_eaten = self._update()

        # Observations
        observations = self._get_observations()
        assert not np.any(np.isnan(observations))
        # Rewards
        if self.done:
            reward = -1.0
        elif len([p for p in self.state["particles"] if p["active"]]) == 0:
            reward = 1.0
            self.done = True
        elif particles_eaten > 0:
            reward = 0.5
        else:
            reward = -0.005
            # distances_robot_particle = []
            # for robot in self.state["robots"]:
            #     if not robot["active"]:
            #         continue
            #     for particle in self.state["particles"]:
            #         if not particle["active"]:
            #             continue
            #         distances_robot_particle.append(
            #             math.sqrt(
            #                 pow(robot["position"]["x"] - particle["position"]["x"], 2)
            #                 + pow(robot["position"]["y"] - particle["position"]["y"], 2)
            #             )
            #         )
            # reward = -(
            #     (
            #         min(distances_robot_particle)
            #         / math.sqrt(pow(self.world_width, 2) + pow(self.world_height, 2))
            #     )
            #     / 10
            #     if distances_robot_particle
            #     else 0
            # )

        # Terminated
        terminated = False
        if self.state["time"] >= self.max_episode_time:
            terminated = True
        # Info
        info = {}
        return observations, reward, terminated, self.done, info

    def reset(self, *, seed=None, options=None):
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
        return self._get_observations(), {}

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
                (100, 100, 255),
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
                    "colliding": False,
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
                    "explosionTimes": [[self.max_episode_time + 1]],
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
            offset = self.max_robots * 8 + i * 4
            obs[offset] = self._normalize(
                particle["position"]["x"], 0, self.world_width
            )
            obs[offset + 1] = self._normalize(
                particle["position"]["y"], 0, self.world_height
            )
            obs[offset + 2] = self._normalize(
                particle["radius"], self.min_particle_radius, self.max_particle_radius
            )
            obs[offset + 3] = particle["active"]

        offset = self.max_robots * 8 + self.max_particles * 4
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
        obs[offset + 3 : offset + 3 + self.max_robots + 1] = self._one_hot(
            self.state["num_active_robots"], self.max_robots + 1
        )
        offset = offset + 3 + self.max_robots + 1
        obs[offset:] = self._one_hot(
            self.state["num_active_particles"], self.max_particles + 1
        )
        return obs

    def _update(self) -> int:
        elapsed_time = 0
        particles_eaten = 0
        while elapsed_time < self.command_time_interval:
            particles_eaten += self._check_collisions()
            self._update_robots()
            # self._update_particles()
            self.state["time"] += self.dt
            elapsed_time += self.dt

        return particles_eaten

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

    def _check_collisions(self) -> int:
        particles_eaten = 0
        for i, robot in enumerate(self.state["robots"]):
            robot["colliding"] = False
            if not robot["active"]:
                continue
            for j, particle in enumerate(self.state["particles"]):
                if not particle["active"]:
                    continue
                if self._is_collision(robot, particle):
                    robot["colliding"] = True
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
                        particule = self.state["particles"].pop(j)
                        self.state["robots"][i]["score"] += particule["radius"]
                        particles_eaten += 1
                    else:
                        robot_next = deepcopy(robot)
                        self._update_robot(robot_next)
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
                    robot["colliding"] = True
                    other_robot["colliding"] = True
                    robot_next = deepcopy(robot)
                    other_robot_next = deepcopy(other_robot)
                    self._update_robot(robot_next)
                    self._update_robot(robot_next)
                    self._update_robot(other_robot_next)
                    self._update_robot(other_robot_next)
                    if self._is_collision(robot_next, other_robot_next):
                        if robot["leftSpeed"] + robot["rightSpeed"] != 0:
                            robot["leftSpeed"] = 0
                            robot["rightSpeed"] = 0
                        if other_robot["leftSpeed"] + other_robot["rightSpeed"] != 0:
                            other_robot["leftSpeed"] = 0
                            other_robot["rightSpeed"] = 0
        return particles_eaten

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
