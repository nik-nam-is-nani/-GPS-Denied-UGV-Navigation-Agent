import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, nfeatures=500, scale_factor=0.05, seed=None):
        self.nfeatures = nfeatures
        self.scale_factor = scale_factor
        self.rng = np.random.default_rng(seed)

        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev_kp = None
        self.prev_des = None
        self.prev_frame = None

        self.min_matches = 10
        self.max_matches = 50

    def estimate(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return np.zeros(2), 0.0

        kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
        kp2, des2 = self.orb.detectAndCompute(curr_frame, None)

        if des1 is None or des2 is None:
            return np.zeros(2), 0.0

        if len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            return np.zeros(2), 0.0

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < self.min_matches:
            return np.zeros(2), 0.0

        matches = matches[:self.max_matches]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        delta = np.mean(pts2 - pts1, axis=0)

        confidence = min(len(matches) / 50.0, 1.0)

        if confidence < 0.3:
            return np.zeros(2), confidence

        delta_world = delta * self.scale_factor

        return delta_world, confidence

    def process_frame(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.zeros(2), 0.0

        delta, confidence = self.estimate(self.prev_frame, frame)
        self.prev_frame = frame

        return delta, confidence


class MonoVisualOdometry:
    def __init__(self, focal_length=500, baseline=0.5, scale_factor=0.05, seed=None):
        self.focal_length = focal_length
        self.baseline = baseline
        self.scale_factor = scale_factor
        self.rng = np.random.default_rng(seed)

        self.vo = VisualOdometry(nfeatures=500, scale_factor=scale_factor, seed=seed)

    def compute_depth(self, left_frame, right_frame):
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        disparity = stereo.compute(left_frame, right_frame)

        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_mask = disparity > 0
        depth[valid_mask] = (self.focal_length * self.baseline) / disparity[valid_mask]

        return depth

    def estimate(self, prev_frame, curr_frame):
        return self.vo.estimate(prev_frame, curr_frame)


def create_test_frames(map_grid, pos, heading, view_size=80):
    h, w = view_size, view_size
    frame = np.ones((h, w, 3), dtype=np.uint8) * 200

    center = (w // 2, h // 2)
    cos_h, sin_h = np.cos(heading), np.sin(heading)

    for y in range(h):
        for x in range(w):
            world_x = pos[0] + (x - center[0]) * cos_h - (y - center[1]) * sin_h
            world_y = pos[1] + (x - center[0]) * sin_h + (y - center[1]) * cos_h

            if 0 <= world_x < map_grid.shape[1] and 0 <= world_y < map_grid.shape[0]:
                if map_grid[int(world_y), int(world_x)] == 1:
                    shade = np.random.randint(30, 80)
                    frame[y, x] = [shade, shade, shade]

    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)