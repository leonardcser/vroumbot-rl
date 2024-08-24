import time

import numpy as np
import pygame

from robot_particle_env import RobotParticleEnv


def main():
    pygame.init()
    env = RobotParticleEnv()
    env.reset()

    try:
        while True:
            env.render(mode="human")
            left_speed = 0
            right_speed = 0
            if pygame.key.get_focused():
                keys = pygame.key.get_pressed()
                if keys[pygame.K_a]:
                    right_speed = env.max_forward_speed
                if keys[pygame.K_d]:
                    left_speed = env.max_forward_speed
                if keys[pygame.K_w]:
                    left_speed = env.max_forward_speed
                    right_speed = env.max_forward_speed
                if keys[pygame.K_s]:
                    left_speed = -env.max_backward_speed
                    right_speed = -env.max_backward_speed
                if keys[pygame.K_q]:
                    left_speed = -env.max_backward_speed
                    right_speed = env.max_backward_speed
                if keys[pygame.K_e]:
                    left_speed = env.max_backward_speed
                    right_speed = -env.max_backward_speed
            out = env.step(np.array([left_speed, right_speed]))
            print(out[0])
            if out[2]:
                env.reset()
            time.sleep(0.1)  # Adjust the sleep time as needed
    except KeyboardInterrupt:
        env.close()


if __name__ == "__main__":
    main()
