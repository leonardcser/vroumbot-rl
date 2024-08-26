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
            velocity = 0.5
            angle = 0.5
            if pygame.key.get_focused():
                keys = pygame.key.get_pressed()
                if keys[pygame.K_a]:
                    angle = 1
                if keys[pygame.K_d]:
                    angle = 0
                if keys[pygame.K_w]:
                    velocity = 1
                if keys[pygame.K_s]:
                    velocity = 0
            out = env.step(np.array([velocity, angle]))
            print(out[0])
            if out[2] or out[3]:
                env.reset()
    except KeyboardInterrupt:
        env.close()


if __name__ == "__main__":
    main()
