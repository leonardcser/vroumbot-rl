import os
import tempfile
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pygame
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger

from robot_particle_env import RobotParticleEnv


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def main(args):
    checkpoint_path = Path.cwd() / "checkpoint"

    if args.train or args.resume:
        if args.resume:
            algo = Algorithm.from_checkpoint(checkpoint_path)
        else:
            config = (
                PPOConfig()
                .environment(RobotParticleEnv)
                .env_runners(num_env_runners=1)
                .framework("torch")
                .training(
                    gamma=0.9,
                    lr=0.0001,
                    kl_coeff=0.2,
                    # entropy_coeff=0.001,
                    model={
                        "fcnet_hiddens": [256, 256, 256, 512],
                        "fcnet_activation": "relu",
                    },
                )
            )

            algo = config.build(
                logger_creator=custom_log_creator(Path.cwd() / "logs", "ppo")
            )

        for i in range(50):
            result = algo.train()
            result.pop("config")
            # print important metrics
            metrics = result["env_runners"]
            print("-" * 50)
            print("Iteration:", i)
            print("Mean Length:", metrics["episode_len_mean"])
            print("Min Reward:", metrics["episode_reward_min"])
            print("Max Reward:", metrics["episode_reward_max"])
            print("Mean Reward:", metrics["episode_reward_mean"])

        # save the model to disk
        checkpoint_path.mkdir(exist_ok=True)
        algo.save_checkpoint(checkpoint_path)
        print("Checkpoint saved to", checkpoint_path)

    else:
        algo = Algorithm.from_checkpoint(checkpoint_path)

        pygame.init()
        env = RobotParticleEnv()
        obs, _ = env.reset()

        while True:
            action = algo.compute_single_action(obs)
            obs, reward, terminated, done, info = env.step(action)
            env.render()
            if done or terminated:
                obs, _ = env.reset()
            time.sleep(0.05)


if __name__ == "__main__":
    argparse = ArgumentParser()
    # train and resume subcommands
    argparse.add_argument("--train", action="store_true")
    argparse.add_argument("--resume", action="store_true")
    args = argparse.parse_args()
    main(args)
