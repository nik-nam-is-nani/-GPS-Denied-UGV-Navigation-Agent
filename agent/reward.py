import numpy as np


def compute_reward(pos, prev_pos, goal, estimated_pos, collided, reached_goal,
                   vo_confidence=1.0, step_penalty=0.1):
    reward = 0.0

    prev_dist = np.linalg.norm(prev_pos - goal)
    curr_dist = np.linalg.norm(pos - goal)

    progress = prev_dist - curr_dist
    reward += progress * 2.0

    if reached_goal:
        reward += 100.0

    if collided:
        reward -= 50.0

    drift = np.linalg.norm(pos - estimated_pos)
    drift_penalty = drift * 0.1

    if vo_confidence < 0.5:
        drift_penalty *= 1.5

    reward -= drift_penalty

    reward += step_penalty

    return reward


def compute_reward_curriculum(stage, **kwargs):
    progress_weight = {
        1: 1.0,
        2: 1.5,
        3: 2.0,
        4: 2.5
    }.get(stage, 2.0)

    reward = compute_reward(**kwargs)

    if stage >= 3:
        gps_penalty = kwargs.get('gps_drift', 0) * 0.2
        reward -= gps_penalty

    return reward


class RewardLogger:
    def __init__(self):
        self.episode_rewards = []
        self.current_episode = []
        self.stats = {
            'progress': [],
            'drift': [],
            'collision': [],
            'goal_reached': []
        }

    def add_step(self, reward, info):
        self.current_episode.append(reward)

        self.stats['progress'].append(-info.get('distance_to_goal', 0))
        self.stats['drift'].append(info.get('drift', 0))
        self.stats['collision'].append(1 if info.get('collided', False) else 0)
        self.stats['goal_reached'].append(1 if info.get('reached_goal', False) else 0)

    def end_episode(self):
        self.episode_rewards.append(sum(self.current_episode))
        self.current_episode = []

    def get_stats(self):
        if not self.episode_rewards:
            return {}

        return {
            'mean_reward': np.mean(self.episode_rewards[-10:]),
            'mean_progress': np.mean(self.stats['progress'][-100:]),
            'mean_drift': np.mean(self.stats['drift'][-100:]),
            'collision_rate': np.mean(self.stats['collision'][-100:]),
            'goal_rate': np.mean(self.stats['goal_reached'][-100:])
        }


def create_reward_function(stage=1):
    def reward_fn(pos, prev_pos, goal, estimated_pos, collided, reached_goal,
                  vo_confidence=1.0):
        return compute_reward_curriculum(stage,
            pos=pos, prev_pos=prev_pos, goal=goal,
            estimated_pos=estimated_pos, collided=collided,
            reached_goal=reached_goal, vo_confidence=vo_confidence)

    return reward_fn