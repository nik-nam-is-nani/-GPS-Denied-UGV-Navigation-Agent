import numpy as np

TILE_EMPTY = 0
TILE_WALL  = 1

class GridMap:
    def __init__(self, cols=40, rows=30, tile=20):
        self.cols = cols
        self.rows = rows
        self.tile = tile          # pixels per cell
        self.width  = cols * tile
        self.height = rows * tile
        self.grid = np.zeros((rows, cols), dtype=np.uint8)
        self._build()

    def _build(self):
        g = self.grid
        # border walls
        g[0, :]  = TILE_WALL
        g[-1, :] = TILE_WALL
        g[:, 0]  = TILE_WALL
        g[:, -1] = TILE_WALL
        # interior obstacles (hand-placed for a nice first demo)
        obstacles = [
            (5,  3, 1,  8),   # (col, row, w, h) in tiles
            (15, 8, 8,  1),
            (10, 15, 1, 8),
            (25, 5, 1,  10),
            (30, 18, 8, 1),
            (20, 22, 6, 1),
        ]
        for col, row, w, h in obstacles:
            g[row:row+h, col:col+w] = TILE_WALL

    def is_wall(self, col, row):
        r = int(row)
        c = int(col)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True
        return self.grid[r, c] == TILE_WALL

    def world_to_tile(self, px, py):
        return px // self.tile, py // self.tile

    def tile_to_world_center(self, col, row):
        return col * self.tile + self.tile // 2, row * self.tile + self.tile // 2