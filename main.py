import os
import tempfile
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from pprint import pprint

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger

from robot_particle_env import RobotParticleEnv, StackedRobotParticleEnv

checkpoint_path = Path.cwd() / "checkpoint"


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def train(resume: bool):
    if resume:
        algo = Algorithm.from_checkpoint(checkpoint_path)
    else:
        config = (
            PPOConfig()
            .environment(RobotParticleEnv)
            .env_runners(num_env_runners=1)
            .framework("torch")
            .training(
                gamma=0.99,
                lr=0.00025,
                kl_coeff=0.2,
                entropy_coeff=0.001,
                model={
                    "fcnet_hiddens": [256, 256, 512, 512],
                    "fcnet_activation": "relu",
                },
            )
        )

        algo = config.build(
            logger_creator=custom_log_creator(Path.cwd() / "logs", "ppo")
        )

    checkpoint_path.mkdir(exist_ok=True)
    try:
        for i in range(500):
            result = algo.train()
            result.pop("config")
            pprint(result)
            # print("-" * 50)
            # print("Iteration:", i)
            # print("Mean Length:", metrics["episode_len_mean"])
            # print("Min Reward:", metrics["episode_reward_min"])
            # print("Max Reward:", metrics["episode_reward_max"])
            # print("Mean Reward:", metrics["episode_reward_mean"])

            if i % 10 == 0:
                algo.save_checkpoint(checkpoint_path)
                print("Checkpoint saved to", checkpoint_path)
    except KeyboardInterrupt:
        pass
    finally:
        # save the model to disk
        algo.save_checkpoint(checkpoint_path)
        print("Checkpoint saved to", checkpoint_path)


def tune():
    config = (
        PPOConfig()
        .environment(RobotParticleEnv)
        .training(
            gamma=0.9,
            lr=ray.tune.grid_search([0.001, 0.0001, 0.00001]),
            kl_coeff=0.2,
            entropy_coeff=ray.tune.grid_search([0.0, 0.001, 0.01]),
            model={
                "fcnet_hiddens": ray.tune.grid_search(
                    [[256, 256, 256, 256], [512, 512, 512, 512], [512, 512]]
                ),
                "fcnet_activation": "relu",
            },
        )
    )

    tuner = ray.tune.Tuner(
        "PPO",
        param_space=config,
        run_config=ray.train.RunConfig(
            # stop={"env_runners/episode_return_mean": 150.0},
        ),
    )
    try:
        tuner.fit()
    except KeyboardInterrupt:
        pass
    finally:
        # save the best model to disk
        algo = tuner.get_best_model()
        algo.save_checkpoint(checkpoint_path)
        print("Checkpoint saved to", checkpoint_path)


def run():
    algo = Algorithm.from_checkpoint(checkpoint_path)

    env = RobotParticleEnv(env_config=dict(render_mode="human"))
    obs, _ = env.reset()

    while True:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, done, info = env.step(action)
        print("Reward:", reward)
        if done or terminated:
            obs, _ = env.reset()


def main():
    parser = ArgumentParser(
        description="Robot Particle Environment Training and Running"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    # Tune subcommand
    subparsers.add_parser("tune", help="Tune hyperparameters")

    # Run subcommand
    subparsers.add_parser("run", help="Run the trained model")

    args = parser.parse_args()

    if args.command == "train":
        train(args.resume)
    elif args.command == "tune":
        tune()
    elif args.command == "run":
        run()


if __name__ == "__main__":
    main()
