import heapq
import math
from map import GridMap

def heuristic(a, b):
    """Euclidean distance heuristic."""
    return math.hypot(b[0] - a[0], b[1] - a[1])

def get_neighbors(node, grid_map):
    """Returns valid neighboring tile coords (4-connected)."""
    col, row = node
    candidates = [
        (col + 1, row),
        (col - 1, row),
        (col, row + 1),
        (col, row - 1),
    ]
    neighbors = []
    for c, r in candidates:
        if not grid_map.is_wall(c, r):
            neighbors.append((c, r))
    return neighbors

def astar(grid_map, start_tile, goal_tile):
    """
    A* search on a GridMap.
    start_tile, goal_tile: (col, row) tile coordinates.
    Returns: list of (col, row) from start to goal, or [] if no path.
    """
    if grid_map.is_wall(goal_tile[0], goal_tile[1]):
        return []

    open_set = []
    heapq.heappush(open_set, (0, start_tile))
    came_from = {}
    g_score = {start_tile: 0}
    f_score = {start_tile: heuristic(start_tile, goal_tile)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_tile:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for neighbor in get_neighbors(current, grid_map):
            tentative_g = g_score[current] + 1  # uniform cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal_tile)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []  # no path found
