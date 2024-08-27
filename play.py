import numpy as np
import pygame

from robot_particle_env import StackedRobotParticleEnv


def main():
    pygame.init()
    env = StackedRobotParticleEnv(env_config=dict(render_mode="human"))
    env.reset()
    env = env.env

    try:
        while True:
            left_speed = 0.5
            right_speed = 0.5
            if pygame.key.get_focused():
                keys = pygame.key.get_pressed()
                if keys[pygame.K_a]:
                    right_speed = 1
                if keys[pygame.K_d]:
                    left_speed = 1
                if keys[pygame.K_w]:
                    left_speed = 1
                    right_speed = 1
                if keys[pygame.K_s]:
                    left_speed = 0
                    right_speed = 0
                if keys[pygame.K_q]:
                    left_speed = 0
                    right_speed = 1
                if keys[pygame.K_e]:
                    left_speed = 1
                    right_speed = 0
            out = env.step(np.array([left_speed, right_speed]))
            print(out[1])
            if out[2] or out[3]:
                env.reset()
    except KeyboardInterrupt:
        env.close()


if __name__ == "__main__":
    main()
