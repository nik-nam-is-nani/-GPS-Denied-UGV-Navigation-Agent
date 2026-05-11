import numpy as np
import cv2


class TerrainGenerator:
    def __init__(self, map_size=200, seed=None):
        self.map_size = map_size
        self.rng = np.random.default_rng(seed)
        self.grid = np.zeros((map_size, map_size), dtype=np.uint8)

    def generate(self, difficulty="medium"):
        if difficulty == "easy":
            self._open_area()
        elif difficulty == "medium":
            self._walls_and_corridors()
        elif difficulty == "hard":
            self._complex_maze()
        elif difficulty == "featureless":
            self._featureless_desert()
        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        return self.grid

    def _open_area(self):
        pass

    def _walls_and_corridors(self):
        num_walls = self.rng.integers(8, 15)
        for _ in range(num_walls):
            x1 = self.rng.integers(10, self.map_size - 40)
            y1 = self.rng.integers(10, self.map_size - 40)
            length = self.rng.integers(20, 60)
            horizontal = self.rng.random() > 0.5

            if horizontal:
                x2 = x1 + length
                y2 = y1
            else:
                x2 = x1
                y2 = y1 + length

            cv2.line(self.grid, (x1, y1), (x2, y2), 1, thickness=self.rng.integers(2, 5))

    def _complex_maze(self):
        self._walls_and_corridors()
        num_obstacles = self.rng.integers(10, 20)
        for _ in range(num_obstacles):
            cx = self.rng.integers(20, self.map_size - 20)
            cy = self.rng.integers(20, self.map_size - 20)
            radius = self.rng.integers(3, 10)
            cv2.circle(self.grid, (cx, cy), radius, 1, -1)

    def _featureless_desert(self):
        num_dunes = self.rng.integers(3, 8)
        for _ in range(num_dunes):
            cx = self.rng.integers(30, self.map_size - 30)
            cy = self.rng.integers(30, self.map_size - 30)
            radius = self.rng.integers(15, 30)
            cv2.circle(self.grid, (cx, cy), radius, 1, -1)

    def is_collision(self, pos, radius=3):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            return self.grid[y, x] == 1
        return True

    def raycast_lidar(self, pos, heading, num_rays=8, max_range=30):
        angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays) + heading
        distances = []

        for angle in angles:
            step = 0.5
            curr_x, curr_y = pos[0], pos[1]

            for dist in np.arange(0, max_range, step):
                curr_x = pos[0] + dist * np.cos(angle)
                curr_y = pos[1] + dist * np.sin(angle)

                if not (0 <= curr_x < self.map_size and 0 <= curr_y < self.map_size):
                    break

                if self.grid[int(curr_y), int(curr_x)] == 1:
                    break

            distances.append(dist)

        return np.array(distances)

    def render_frame(self, pos, heading, view_distance=40):
        h, w = view_distance * 2, view_distance * 2
        frame = np.ones((h, w, 3), dtype=np.uint8) * 200

        center = (w // 2, h // 2)
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        for y in range(h):
            for x in range(w):
                world_x = pos[0] + (x - center[0]) * cos_h - (y - center[1]) * sin_h
                world_y = pos[1] + (x - center[0]) * sin_h + (y - center[1]) * cos_h

                if 0 <= world_x < self.map_size and 0 <= world_y < self.map_size:
                    if self.grid[int(world_y), int(world_x)] == 1:
                        frame[y, x] = np.array([50, 50, 50])

        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        return frame