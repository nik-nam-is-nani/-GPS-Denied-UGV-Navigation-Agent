import math
import numpy as np
import pygame

class Robot:
    def __init__(self, x, y, heading=0.0):
        self.x       = float(x)
        self.y       = float(y)
        self.heading = float(heading)   # radians; 0 = right
        self.speed   = 0.0
        self.radius  = 8                # pixels, for collision

        # Physics limits
        self.max_speed    = 3.0
        self.accel        = 0.15
        self.friction     = 0.12
        self.turn_speed   = 0.055       # radians/frame

        # True pose (what GPS would give — we'll hide this later)
        self.true_x = x
        self.true_y = y

        # Odometry pose (dead-reckoning estimate — drifts over time)
        self.odo_x   = x
        self.odo_y   = y
        self.odo_hdg = heading
        self.odo_drift_factor = 0.002   # small noise added each step

        # GPS state
        self.gps_mode = "on"   # "on" | "off" | "spoof" | "drift"
        self._spoof_offset = np.array([60.0, 40.0])
        self._drift_accum  = np.zeros(2)

        # Trail history for drawing
        self.true_trail = []
        self.odo_trail  = []

    # ------------------------------------------------------------------ movement
    def update(self, throttle, steer, grid_map):
        """throttle in [-1,1], steer in [-1,1]"""
        # Acceleration / friction
        self.speed += throttle * self.accel
        self.speed *= (1 - self.friction)
        self.speed  = max(-self.max_speed * 0.5,
                          min(self.max_speed, self.speed))

        # Heading
        if abs(self.speed) > 0.05:
            self.heading += steer * self.turn_speed * (1 if self.speed > 0 else -1)

        # Proposed new position
        dx = self.speed * math.cos(self.heading)
        dy = self.speed * math.sin(self.heading)
        nx = self.x + dx
        ny = self.y + dy

        # Tile collision — axis-separated so robot slides along walls
        tc, tr = grid_map.world_to_tile(nx, self.y)
        if not grid_map.is_wall(tc, tr):
            self.x = nx
        else:
            self.speed *= -0.3   # bounce slightly

        tc, tr = grid_map.world_to_tile(self.x, ny)
        if not grid_map.is_wall(tc, tr):
            self.y = ny
        else:
            self.speed *= -0.3

        self.true_x, self.true_y = self.x, self.y

        # Dead-reckoning odometry (accumulates error)
        noise = np.random.randn(2) * self.odo_drift_factor * max(0.1, abs(self.speed))
        self.odo_x   += dx + noise[0]
        self.odo_y   += dy + noise[1]
        self.odo_hdg  = self.heading + np.random.randn() * 0.001

        # History
        self.true_trail.append((self.true_x, self.true_y))
        self.odo_trail.append((self.odo_x, self.odo_y))
        if len(self.true_trail) > 300:
            self.true_trail.pop(0)
            self.odo_trail.pop(0)

    # ------------------------------------------------------------------ rendering
    def draw(self, surf, shake_x=0, shake_y=0):
        """Draw the robot (true position) with a direction arrow."""
        cx = int(self.x) + shake_x
        cy = int(self.y) + shake_y
        pygame.draw.circle(surf, ( 0, 170, 255), (cx, cy), self.radius)
        pygame.draw.circle(surf, (255,255,255),        (cx, cy), self.radius, 2)
        ex = cx + int(16 * math.cos(self.heading))
        ey = cy + int(16 * math.sin(self.heading))
        pygame.draw.line(surf, (255,255,255), (cx, cy), (ex, ey), 3)

    # ------------------------------------------------------------------ sensors
    def cast_rays(self, grid_map, num_rays=8, max_dist=150):
        """Returns list of (hit_x, hit_y, dist) for each ray."""
        results = []
        for i in range(num_rays):
            angle = self.heading + (2 * math.pi * i / num_rays)
            dist  = 0
            hit_x, hit_y = self.x, self.y
            while dist < max_dist:
                dist += 2
                hit_x = self.x + dist * math.cos(angle)
                hit_y = self.y + dist * math.sin(angle)
                tc, tr = grid_map.world_to_tile(hit_x, hit_y)
                if grid_map.is_wall(tc, tr):
                    break
            results.append((hit_x, hit_y, dist))
        return results