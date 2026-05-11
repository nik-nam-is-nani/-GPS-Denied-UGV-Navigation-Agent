import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

from env.ugv_env import UGVNavEnv


def make_env(map_size=200, max_steps=500, gps_denied=True, difficulty="medium", seed=None):
    def _init():
        env = UGVNavEnv(
            map_size=map_size,
            max_steps=max_steps,
            gps_denied=gps_denied,
            difficulty=difficulty,
            seed=seed
        )
        env = Monitor(env)
        return env
    return _init


def train_stage(stage_config):
    map_size = stage_config['map_size']
    gps_denied = stage_config['gps_denied']
    difficulty = stage_config['difficulty']
    total_steps = stage_config['steps']
    pretrained_path = stage_config.get('pretrained', None)

    vec_env = DummyVecEnv([
        make_env(map_size=map_size, gps_denied=gps_denied, difficulty=difficulty, seed=i)
        for i in range(8)
    ])

    vec_env = VecMonitor(vec_env)

    if pretrained_path:
        model = PPO.load(pretrained_path, env=vec_env)
        print(f"Loaded pretrained model from {pretrained_path}")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"./tensorboard/stage_{stage_config['name']}/",
            device="auto"
        )

    eval_env = DummyVecEnv([make_env(map_size=map_size, gps_denied=gps_denied, difficulty=difficulty)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/stage_{stage_config['name']}/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./checkpoints/stage_{stage_config['name']}/",
        name_prefix="ugv_model"
    )

    model.learn(
        total_timesteps=total_steps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save(f"./models/ugv_stage_{stage_config['name']}")

    return model


def curriculum_training():
    stages = [
        {
            'name': '1',
            'map_size': 50,
            'gps_denied': False,
            'difficulty': 'easy',
            'steps': 200000
        },
        {
            'name': '2',
            'map_size': 100,
            'gps_denied': False,
            'difficulty': 'medium',
            'steps': 500000,
            'pretrained': './models/ugv_stage_1.zip'
        },
        {
            'name': '3',
            'map_size': 200,
            'gps_denied': True,
            'difficulty': 'medium',
            'steps': 800000,
            'pretrained': './models/ugv_stage_2.zip'
        },
        {
            'name': '4',
            'map_size': 200,
            'gps_denied': True,
            'difficulty': 'hard',
            'steps': 500000,
            'pretrained': './models/ugv_stage_3.zip'
        }
    ]

    for stage in stages:
        print(f"\n{'='*50}")
        print(f"Starting Stage {stage['name']}")
        print(f"Map size: {stage['map_size']}, GPS denied: {stage['gps_denied']}, Difficulty: {stage['difficulty']}")
        print(f"{'='*50}\n")

        model = train_stage(stage)

    print("\nTraining complete!")
    return model


def quick_test():
    env = UGVNavEnv(map_size=200, max_steps=500, gps_denied=True, difficulty="medium", seed=42)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=10000, progress_bar=True)
    model.save("./models/ugv_quick_test")

    print("Quick test complete!")


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./tensorboard", exist_ok=True)

    print("Choose training mode:")
    print("1. Quick test (10k steps)")
    print("2. Full curriculum training")
    print("3. Single stage")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        quick_test()
    elif choice == "2":
        curriculum_training()
    elif choice == "3":
        stage = {
            'name': 'test',
            'map_size': 100,
            'gps_denied': True,
            'difficulty': 'medium',
            'steps': 50000
        }
        train_stage(stage)
    else:
        print("Invalid choice, running quick test")
        quick_test()