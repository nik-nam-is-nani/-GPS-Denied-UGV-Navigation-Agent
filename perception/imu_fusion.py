import numpy as np


class EKFPoseFusion:
    def __init__(self, initial_state=None, process_noise=0.01,
                 vo_noise=0.5, imu_noise=0.05, seed=None):
        self.rng = np.random.default_rng(seed)

        if initial_state is None:
            self.state = np.zeros(4)
        else:
            self.state = initial_state.copy()

        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * process_noise

        self.R_vo = np.eye(2) * vo_noise
        self.R_imu = np.eye(2) * imu_noise

        self.last_update = None

    def predict(self, dt=0.1, steering=0.0, throttle=0.0):
        x, y, vx, vy = self.state

        speed = throttle * 3.0
        acceleration = np.array([
            speed * np.cos(steering * 0.15),
            speed * np.sin(steering * 0.15)
        ])

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.state[0] += vx * dt + 0.5 * acceleration[0] * dt**2
        self.state[1] += vy * dt + 0.5 * acceleration[1] * dt**2
        self.state[2] += acceleration[0] * dt
        self.state[3] += acceleration[1] * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_vo(self, pos_measurement):
        if pos_measurement is None:
            return

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = pos_measurement
        z = z + self.rng.normal(0, 0.1, 2)

        self._update(z, H, self.R_vo)

    def update_imu(self, vel_measurement):
        if vel_measurement is None:
            return

        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        z = vel_measurement
        z = z + self.rng.normal(0, 0.05, 2)

        self._update(z, H, self.R_imu)

    def _update(self, z, H, R):
        y = z - H @ self.state

        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.P = (self.P + self.P.T) / 2

    def get_position(self):
        return self.state[:2].copy()

    def get_velocity(self):
        return self.state[2:].copy()

    def get_covariance(self):
        return self.P.copy()


class IMUSimulator:
    def __init__(self, noise_scale=0.05, bias_stability=0.001, seed=None):
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale
        self.bias_stability = bias_stability

        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)

    def read_gyro(self, true_angular_vel):
        noise = self.rng.normal(0, self.noise_scale, 3)
        self.gyro_bias += self.rng.normal(0, self.bias_stability, 3)
        return true_angular_vel + noise + self.gyro_bias

    def read_accel(self, true_acceleration):
        noise = self.rng.normal(0, self.noise_scale * 0.5, 3)
        self.accel_bias += self.rng.normal(0, self.bias_stability, 3)
        gravity = np.array([0, 0, 9.81])
        return true_acceleration + noise + self.accel_bias + gravity

    def get_velocity_estimate(self, true_velocity):
        noise = self.rng.normal(0, self.noise_scale * 2, 2)
        return true_velocity + noise


def create_ekf_fusion(vo_enabled=True, imu_enabled=True):
    ekf = EKFPoseFusion(process_noise=0.01, vo_noise=0.5, imu_noise=0.05)
    return ekf