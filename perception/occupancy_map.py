import numpy as np
import cv2


class OccupancyMap:
    def __init__(self, map_size=200, resolution=1.0, seed=None):
        self.map_size = map_size
        self.resolution = resolution

        self.grid = np.zeros((map_size, map_size), dtype=np.float32)
        self.visited = np.zeros((map_size, map_size), dtype=np.uint8)

        self.log_odds_free = -1.0
        self.log_odds_occ = 1.0
        self.log_odds_prior = 0.0

        self.rng = np.random.default_rng(seed)

    def update(self, position, heading, measurements, max_range=30):
        px, py = int(position[0]), int(position[1])

        if not (0 <= px < self.map_size and 0 <= py < self.map_size):
            return

        self.visited[py, px] = 1

        num_rays = len(measurements)
        fov = np.pi
        angles = np.linspace(heading - fov / 2, heading + fov / 2, num_rays)

        for angle, distance in zip(angles, measurements):
            if distance >= max_range:
                continue

            for r in np.arange(0, distance, self.resolution):
                ex = px + r * np.cos(angle)
                ey = py + r * np.sin(angle)

                if 0 <= ex < self.map_size and 0 <= ey < self.map_size:
                    self._update_cell(int(ex), int(ey), -0.5)

            end_x = px + distance * np.cos(angle)
            end_y = py + distance * np.sin(angle)

            if 0 <= end_x < self.map_size and 0 <= end_y < self.map_size:
                self._update_cell(int(end_x), int(end_y), 0.8)

    def _update_cell(self, x, y, log_odds_increment):
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            current_log_odds = self.log_odds_to_prob(self.grid[y, x])
            new_log_odds = np.log(current_log_odds / (1 - current_log_odds + 1e-10))
            new_log_odds += log_odds_increment

            new_prob = self.log_odds_to_prob(new_log_odds)
            self.grid[y, x] = np.clip(new_prob, 0.01, 0.99)

    def log_odds_to_prob(self, log_odds):
        return 1 - 1 / (1 + np.exp(log_odds))

    def is_free(self, x, y, threshold=0.3):
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            return self.grid[y, x] < threshold
        return False

    def is_occupied(self, x, y, threshold=0.7):
        if 0 <= x < self.map_size and 0 <= y < self.map_size:
            return self.grid[y, x] > threshold
        return False

    def get_binary_map(self, threshold=0.5):
        return (self.grid > threshold).astype(np.uint8)

    def get_visualization(self):
        vis = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        for y in range(self.map_size):
            for x in range(self.map_size):
                prob = self.grid[y, x]

                if self.visited[y, x]:
                    if prob > 0.7:
                        vis[y, x] = [100, 100, 100]
                    elif prob < 0.3:
                        vis[y, x] = [255, 255, 255]
                    else:
                        vis[y, x] = [180, 180, 180]
                else:
                    vis[y, x] = [0, 0, 0]

        return vis


class SLAMIntegrator:
    def __init__(self, map_size=200, resolution=1.0, seed=None):
        self.occupancy_map = OccupancyMap(map_size, resolution, seed)
        self.feature_map = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def update(self, estimated_pos, heading, depth_measurements):
        self.occupancy_map.update(estimated_pos, heading, depth_measurements)

    def get_map(self):
        return self.occupancy_map.grid.copy()

    def get_visualization(self):
        return self.occupancy_map.get_visualization()

    def get_frontier_cells(self):
        frontier = []
        grid = self.occupancy_map.grid

        for y in range(1, self.occupancy_map.map_size - 1):
            for x in range(1, self.occupancy_map.map_size - 1):
                if grid[y, x] < 0.3 and self.occupancy_map.visited[y, x]:
                    free_neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if grid[ny, nx] < 0.3:
                                free_neighbors += 1

                    if free_neighbors < 8:
                        frontier.append((x, y))

        return frontier