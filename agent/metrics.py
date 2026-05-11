import numpy as np
import json
import os


class NavigationMetrics:
    def __init__(self):
        self.episodes = []

    def track_episode(self, true_trajectory, estimated_trajectory, goal,
                     reached_goal, collided, gps_mode="none"):
        if len(true_trajectory) < 2 or len(estimated_trajectory) < 2:
            return None

        true_arr = np.array(true_trajectory)
        est_arr = np.array(estimated_trajectory)

        ate = self._compute_ate(true_arr, est_arr)

        rpe = self._compute_rpe(true_arr, est_arr)

        path_length = self._compute_path_length(true_arr)
        optimal_length = np.linalg.norm(true_arr[0] - true_arr[-1])
        efficiency = optimal_length / path_length if path_length > 0 else 0

        drift_final = np.linalg.norm(true_arr[-1] - est_arr[-1])

        episode_data = {
            'ate': float(ate),
            'rpe': float(rpe),
            'success': reached_goal,
            'collision': collided,
            'path_length': float(path_length),
            'efficiency': float(efficiency),
            'final_drift': float(drift_final),
            'steps': len(true_trajectory),
            'gps_mode': gps_mode
        }

        self.episodes.append(episode_data)
        return episode_data

    def _compute_ate(self, true_trajectory, est_trajectory):
        min_len = min(len(true_trajectory), len(est_trajectory))
        if min_len == 0:
            return float('inf')

        errors = np.linalg.norm(
            true_trajectory[:min_len] - est_trajectory[:min_len],
            axis=1
        )
        return np.mean(errors)

    def _compute_rpe(self, true_trajectory, est_trajectory):
        if len(true_trajectory) < 3:
            return 0.0

        true_deltas = np.diff(true_trajectory, axis=0)
        est_deltas = np.diff(est_trajectory, axis=0)

        min_len = min(len(true_deltas), len(est_deltas))
        if min_len == 0:
            return 0.0

        errors = np.linalg.norm(
            true_deltas[:min_len] - est_deltas[:min_len],
            axis=1
        )
        return np.mean(errors)

    def _compute_path_length(self, trajectory):
        return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

    def get_summary(self, gps_mode=None):
        if not self.episodes:
            return {}

        filtered = self.episodes
        if gps_mode:
            filtered = [e for e in self.episodes if e['gps_mode'] == gps_mode]

        if not filtered:
            return {}

        return {
            'num_episodes': len(filtered),
            'success_rate': np.mean([e['success'] for e in filtered]),
            'collision_rate': np.mean([e['collision'] for e in filtered]),
            'mean_ate': np.mean([e['ate'] for e in filtered]),
            'mean_rpe': np.mean([e['rpe'] for e in filtered]),
            'mean_efficiency': np.mean([e['efficiency'] for e in filtered]),
            'mean_final_drift': np.mean([e['final_drift'] for e in filtered]),
            'std_ate': np.std([e['ate'] for e in filtered])
        }

    def save_results(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                'episodes': self.episodes,
                'summary': self.get_summary()
            }, f, indent=2)

    def compare_gps_modes(self):
        modes = ['none', 'jam', 'spoof', 'drift']
        results = {}

        for mode in modes:
            summary = self.get_summary(gps_mode=mode)
            if summary:
                results[mode] = summary

        return results


def run_evaluation(env, model, num_episodes=20, gps_mode="jam"):
    metrics = NavigationMetrics()

    for i in range(num_episodes):
        obs, _ = env.reset()
        true_traj = []
        est_traj = []

        done = False
        steps = 0

        while not done and steps < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            true_traj.append(info['true_pos'].copy())
            est_traj.append(info['estimated_pos'].copy())

            done = terminated or truncated
            steps += 1

        episode_data = metrics.track_episode(
            true_traj, est_traj, env.goal,
            reached_goal=terminated,
            collided=info.get('collided', False),
            gps_mode=gps_mode
        )

        if episode_data:
            print(f"Episode {i+1}: ATE={episode_data['ate']:.2f}, "
                  f"Success={episode_data['success']}, "
                  f"Drift={episode_data['final_drift']:.2f}")

    return metrics


def print_comparison_table(metrics):
    results = metrics.compare_gps_modes()

    print("\n" + "="*70)
    print("GPS Mode Comparison")
    print("="*70)
    print(f"{'Mode':<12} {'Success':<10} {'Mean ATE':<12} {'Mean Drift':<12}")
    print("-"*70)

    for mode, data in results.items():
        print(f"{mode:<12} {data['success_rate']*100:.1f}%      "
              f"{data['mean_ate']:.2f}       {data['mean_final_drift']:.2f}")

    print("="*70)