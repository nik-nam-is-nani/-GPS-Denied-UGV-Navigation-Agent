import numpy as np


class GPSSpoofAttack:
    def __init__(self, mode="none", seed=None):
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.drift_accumulator = np.zeros(2)
        self.spoof_offset = np.zeros(2)

    def set_mode(self, mode):
        valid_modes = ["none", "jam", "spoof", "drift"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.mode = mode

    def corrupt(self, true_pos, step):
        if self.mode == "none":
            return true_pos.copy()

        elif self.mode == "jam":
            return None

        elif self.mode == "spoof":
            if step == 0:
                self.spoof_offset = self.rng.uniform(-50, 50, 2)
            return true_pos + self.spoof_offset + self.rng.normal(0, 2, 2)

        elif self.mode == "drift":
            self.drift_accumulator += self.rng.normal(0.1, 0.05, 2)
            return true_pos + self.drift_accumulator

        else:
            return true_pos.copy()

    def get_status(self):
        return {
            "mode": self.mode,
            "active": self.mode != "none",
            "description": {
                "none": "GPS operating normally",
                "jam": "GPS signal completely jammed - no fix available",
                "spoof": "GPS coordinates replaced with false position",
                "drift": "GPS coordinates slowly drifting from true position"
            }[self.mode]
        }


class GPSDetector:
    def __init__(self, threshold_drift=10.0, threshold_jumps=30.0):
        self.threshold_drift = threshold_drift
        self.threshold_jumps = threshold_jumps
        self.prev_pos = None

    def detect(self, gps_pos, vo_estimate):
        if gps_pos is None:
            return "jammed"

        if self.prev_pos is not None:
            gps_delta = np.linalg.norm(gps_pos - self.prev_pos)
            if gps_delta > self.threshold_jumps:
                return "spoofed"

            if vo_estimate is not None:
                drift = np.linalg.norm(gps_pos - vo_estimate)
                if drift > self.threshold_drift:
                    return "drifted"

        self.prev_pos = gps_pos.copy()
        return "normal"