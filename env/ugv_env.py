import gymnasium as gym
import numpy as np
import cv2

from env.terrain import TerrainGenerator
from env.gps_spoofer import GPSSpoofAttack


class VisualOdometrySim:
    def __init__(self, noise_scale=0.5, drift_rate=0.02, seed=None):
        self.noise_scale = noise_scale
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)
        self.prev_frame = None

    def compute_motion(self, true_delta):
        noisy_delta = true_delta + self.rng.normal(0, self.noise_scale, 2)
        drift = self.rng.normal(0, self.drift_rate, 2)
        return noisy_delta + drift

    def get_confidence(self, speed):
        if speed < 0.1:
            return 0.3
        elif speed < 0.5:
            return 0.7
        else:
            return 0.9


class UGVNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, map_size=200, max_steps=500, gps_denied=True,
                 difficulty="medium", seed=None):
        super().__init__()
        self.map_size = map_size
        self.max_steps = max_steps
        self.gps_denied = gps_denied
        self.difficulty = difficulty

        self.rng = np.random.default_rng(seed)

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self.terrain = TerrainGenerator(map_size, seed)
        self.grid = self.terrain.generate(difficulty)

        self.gps_spoofer = GPSSpoofAttack(mode="jam" if gps_denied else "none", seed=seed)
        self.vo = VisualOdometrySim(seed=seed)
        self.gps_detector = None

        self.occupancy_map = np.zeros((map_size, map_size), dtype=np.uint8)

        self.pos = None
        self.heading = None
        self.goal = None
        self.step_count = None

        self.estimated_pos = None
        self.prev_pos = None
        self.render_buffer = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.pos = np.array([
            self.rng.uniform(20, 50),
            self.rng.uniform(20, 50)
        ])

        self.heading = self.rng.uniform(0, 2 * np.pi)

        margin = max(10, self.map_size // 4)
        min_pos = margin
        max_pos = self.map_size - margin

        self.goal = np.array([
            self.rng.uniform(min_pos, max_pos),
            self.rng.uniform(min_pos, max_pos)
        ])

        min_dist = self.map_size // 3
        while np.linalg.norm(self.pos - self.goal) < min_dist:
            self.goal = np.array([
                self.rng.uniform(min_pos, max_pos),
                self.rng.uniform(min_pos, max_pos)
            ])

        self.step_count = 0
        self.prev_pos = self.pos.copy()

        self.estimated_pos = self.pos.copy()
        self.render_buffer = self.terrain.render_frame(self.pos, self.heading)

        self.occupancy_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)

        return self._get_obs(), {}

    def step(self, action):
        steer, throttle = action
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)

        speed = throttle * 3.0
        self.heading += steer * 0.15
        self.heading = self.heading % (2 * np.pi)

        true_delta = np.array([
            speed * np.cos(self.heading),
            speed * np.sin(self.heading)
        ])

        self.pos += true_delta
        self.pos = np.clip(self.pos, 0, self.map_size)

        vo_delta = self.vo.compute_motion(true_delta)
        self.estimated_pos += vo_delta

        distances = self.terrain.raycast_lidar(self.pos, self.heading)

        collided = False
        for d in distances:
            if d < 3.0:
                collided = True
                break

        if collided:
            self.pos = self.prev_pos.copy()

        dist_to_goal = np.linalg.norm(self.pos - self.goal)
        reached = dist_to_goal < 5.0

        reward = self._compute_reward(reached, collided, dist_to_goal)

        terminated = reached
        truncated = self.step_count >= self.max_steps

        self.step_count += 1
        self.prev_pos = self.pos.copy()

        self._update_occupancy_map()

        info = {
            "true_pos": self.pos.copy(),
            "estimated_pos": self.estimated_pos.copy(),
            "drift": np.linalg.norm(self.pos - self.estimated_pos),
            "collided": collided,
            "reached_goal": reached
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        dist_to_goal = np.linalg.norm(self.estimated_pos - self.goal)
        goal_angle = np.arctan2(
            self.goal[1] - self.estimated_pos[1],
            self.goal[0] - self.estimated_pos[0]
        ) - self.heading
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        vo_confidence = self.vo.get_confidence(0.5)

        distances = self.terrain.raycast_lidar(self.pos, self.heading)
        distances = distances / 30.0

        obs = np.array([
            self.estimated_pos[0] / self.map_size,
            self.estimated_pos[1] / self.map_size,
            self.heading / (2 * np.pi),
            np.sin(goal_angle),
            np.cos(goal_angle),
            dist_to_goal / self.map_size,
            vo_confidence,
            *distances
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, reached, collided, dist_to_goal):
        reward = 0.0

        if reached:
            reward += 100.0
        elif collided:
            reward -= 50.0

        progress = np.linalg.norm(self.prev_pos - self.goal) - dist_to_goal
        reward += progress * 2.0

        drift = np.linalg.norm(self.pos - self.estimated_pos)
        reward -= drift * 0.1

        reward += 0.1

        return reward

    def _update_occupancy_map(self):
        if 0 <= self.estimated_pos[0] < self.map_size and \
           0 <= self.estimated_pos[1] < self.map_size:
            cx, cy = int(self.estimated_pos[0]), int(self.estimated_pos[1])
            radius = 3

            for y in range(max(0, cy - radius), min(self.map_size, cy + radius + 1)):
                for x in range(max(0, cx - radius), min(self.map_size, cx + radius + 1)):
                    if self.grid[y, x] == 1:
                        self.occupancy_map[y, x] = 1

    def render(self):
        frame = self.terrain.render_frame(self.pos, self.heading)

        scale = 4
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        cv2.circle(frame,
                   (int(self.goal[0] * scale), int(self.goal[1] * scale)),
                   8, (0, 0, 255), -1)

        cv2.putText(frame, f"Step: {self.step_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        drift = np.linalg.norm(self.pos - self.estimated_pos)
        cv2.putText(frame, f"Drift: {drift:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


gym.register("UGVNav-v0", UGVNavEnv)