"""
UGV GPS-Denied Navigation Simulator — Phase 1
Controls:
  W / UP    — accelerate
  S / DOWN  — brake / reverse
  A / LEFT  — steer left
  D / RIGHT — steer right
  G         — cycle GPS mode (on → off → spoof → drift)
  R         — reset robot to start
  ESC       — quit
"""

import pygame, sys, math, time
from map    import GridMap
from robot  import Robot
from astar  import astar

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = (18,  18,  22)
WALL_COL    = (55,  55,  65)
EMPTY_COL   = (28,  28,  35)
ROBOT_COL   = (52,  199, 142)   # green
GOAL_COL    = (255, 200,  50)   # gold
RAY_COL     = (52,  199, 142, 60)
TRUE_TRAIL  = (52,  199, 142, 90)
ODO_TRAIL   = (90,  140, 255, 90)
GPS_COLS    = {
    "on":    (52,  199, 142),
    "off":   (90,   90, 100),
    "spoof": (230,  80,  80),
    "drift": (255, 160,  40),
}
GPS_DOT_RAD = 5
HUD_BG      = (28, 28, 38, 200)

def draw_arrow(surf, colour, x, y, angle, length=16, width=3):
    ex = x + length * math.cos(angle)
    ey = y + length * math.sin(angle)
    pygame.draw.line(surf, colour, (x, y), (ex, ey), width)
    for side in (-0.5, 0.5):
        ax = ex + 8 * math.cos(angle + math.pi + side)
        ay = ey + 8 * math.sin(angle + math.pi + side)
        pygame.draw.line(surf, colour, (ex, ey), (ax, ay), width)

def draw_dashed_line(surf, colour, pts, dash=6, gap=4):
    if len(pts) < 2: return
    acc = 0
    drawing = True
    for i in range(1, len(pts)):
        x0, y0 = pts[i-1]
        x1, y1 = pts[i]
        dx, dy = x1-x0, y1-y0
        seg = math.hypot(dx, dy)
        if seg == 0: continue
        nx, ny = dx/seg, dy/seg
        pos = 0
        while pos < seg:
            end = min(pos + (dash if drawing else gap), seg)
            if drawing:
                pygame.draw.line(surf, colour,
                    (int(x0 + nx*pos), int(y0 + ny*pos)),
                    (int(x0 + nx*end), int(y0 + ny*end)), 1)
            pos = end
            drawing = not drawing

def main():
    pygame.init()
    grid = GridMap(cols=40, rows=30, tile=20)
    W, H  = grid.width, grid.height
    HUD_H = 100
    screen = pygame.display.set_mode((W, H + HUD_H))
    pygame.display.set_caption("UGV GPS-Denied Navigation Simulator")
    clock  = pygame.time.Clock()

    # Surfaces
    map_surf  = pygame.Surface((W, H))
    hud_surf  = pygame.Surface((W, HUD_H), pygame.SRCALPHA)
    ray_surf  = pygame.Surface((W, H),     pygame.SRCALPHA)
    trl_surf  = pygame.Surface((W, H),     pygame.SRCALPHA)
    path_surf = pygame.Surface((W, H),     pygame.SRCALPHA)  # for planned path

    # Pre-draw static map
    for row in range(grid.rows):
        for col in range(grid.cols):
            rect = pygame.Rect(col*grid.tile, row*grid.tile, grid.tile, grid.tile)
            col_c = WALL_COL if grid.grid[row, col] else EMPTY_COL
            pygame.draw.rect(map_surf, col_c, rect)
    # Grid lines
    for r in range(grid.rows+1):
        pygame.draw.line(map_surf, (35,35,42), (0, r*grid.tile), (W, r*grid.tile))
    for c in range(grid.cols+1):
        pygame.draw.line(map_surf, (35,35,42), (c*grid.tile, 0), (c*grid.tile, H))

    # Robot + goal
    start_px, start_py = grid.tile_to_world_center(3, 3)
    goal_px,  goal_py  = grid.tile_to_world_center(35, 25)
    robot = Robot(start_px, start_py, heading=0.0)

    font_big   = pygame.font.SysFont("monospace", 15, bold=True)
    font_small = pygame.font.SysFont("monospace", 12)

    GPS_MODES = ["on", "off", "spoof", "drift"]
    reached = False
    start_time = time.time()
    steps = 0

    # Auto-pilot / pathfinding state
    auto_pilot = False
    path = []           # list of world (x, y) waypoints
    current_wp = 0      # index into path
    goal_px, goal_py = grid.tile_to_world_center(35, 25)

    while True:
        dt = clock.tick(60)

        # ── Events ─────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_g:
                    idx = GPS_MODES.index(robot.gps_mode)
                    robot.gps_mode = GPS_MODES[(idx+1) % len(GPS_MODES)]
                    robot._drift_accum[:] = 0   # reset drift on mode change
                if ev.key == pygame.K_r:
                    robot = Robot(start_px, start_py, heading=0.0)
                    reached = False
                    auto_pilot = False
                    path = []
                    current_wp = 0
                    start_time = time.time()
                    steps = 0
                    path_surf.fill((0,0,0,0))
                    trl_surf.fill((0,0,0,0))
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if my < H:  # ignore clicks on HUD
                    # Convert click to tile
                    tc = mx // grid.tile
                    tr = my // grid.tile
                    if not grid.is_wall(tc, tr):
                        start_tile = grid.world_to_tile(robot.x, robot.y)
                        goal_tile  = (tc, tr)
                        path_tiles = astar(grid, start_tile, goal_tile)
                        if path_tiles:
                            path = [grid.tile_to_world_center(c, r) for c, r in path_tiles]
                            current_wp = 0
                            auto_pilot = True
                            reached = False
                        # clear previous path overlay
                        path_surf.fill((0,0,0,0))

        # ── Input / Auto-pilot ────────────────────────────────────────
        throttle = 0.0
        steer    = 0.0

        if auto_pilot and path and current_wp < len(path):
            wp_x, wp_y = path[current_wp]
            dx = wp_x - robot.x
            dy = wp_y - robot.y
            dist = math.hypot(dx, dy)

            if dist < 10:  # waypoint reached
                current_wp += 1
                if current_wp >= len(path):
                    auto_pilot = False
                    reached = True
            else:
                # Desired heading
                desired = math.atan2(dy, dx)
                # Angle difference (handle wrap-around)
                err = (desired - robot.heading + math.pi) % (2*math.pi) - math.pi
                steer = max(-1.0, min(1.0, err * 3.0))  # P-controller
                throttle = 1.0 if abs(err) < 0.5 else 0.5

        else:
            # Manual control only when not auto-piloting
            if not auto_pilot:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_w] or keys[pygame.K_UP]:    throttle =  1.0
                if keys[pygame.K_s] or keys[pygame.K_DOWN]:  throttle = -1.0
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:  steer    = -1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]: steer    =  1.0

        robot.update(throttle, steer, grid)
        steps += 1

        # Check goal
        dist_to_goal = math.hypot(robot.x - goal_px, robot.y - goal_py)
        if dist_to_goal < 18 and not reached:
            reached = True

        gps = robot.get_gps()
        rays = robot.cast_rays(grid)

        # ── Draw trails (persistent surface) ───────────────────────────
        if len(robot.true_trail) >= 2:
            p = robot.true_trail[-2:]
            pygame.draw.line(trl_surf, (52,199,142, 80),
                             (int(p[0][0]), int(p[0][1])),
                             (int(p[1][0]), int(p[1][1])), 2)
        if len(robot.odo_trail) >= 2:
            p = robot.odo_trail[-2:]
            pygame.draw.line(trl_surf, (90,140,255, 80),
                             (int(p[0][0]), int(p[0][1])),
                             (int(p[1][0]), int(p[1][1])), 1)

        # ── Compose frame ──────────────────────────────────────────────
        screen.blit(map_surf, (0, 0))
        screen.blit(trl_surf, (0, 0))

        # Draw planned A* path
        if path and current_wp < len(path):
            path_surf.fill((0,0,0,0))
            points = [(int(px), int(py)) for px, py in path]
            if len(points) >= 2:
                pygame.draw.lines(path_surf, (255, 200, 50, 180), False, points, 3)
            # highlight current waypoint
            wx, wy = points[current_wp]
            pygame.draw.circle(path_surf, (255, 255, 255), (wx, wy), 6, 2)
            screen.blit(path_surf, (0, 0))

        # Rays
        ray_surf.fill((0,0,0,0))
        for hx, hy, dist in rays:
            col_a = max(0, int(60 * (1 - dist/150)))
            pygame.draw.line(ray_surf, (52,199,142, col_a),
                             (int(robot.x), int(robot.y)),
                             (int(hx), int(hy)), 1)
        screen.blit(ray_surf, (0, 0))

        # Goal
        pygame.draw.circle(screen, GOAL_COL,       (goal_px, goal_py), 10)
        pygame.draw.circle(screen, (255,255,255),   (goal_px, goal_py), 10, 2)
        if reached:
            lbl = font_big.render("GOAL REACHED!", True, GOAL_COL)
            screen.blit(lbl, (goal_px - lbl.get_width()//2, goal_py - 28))

        # GPS dot
        if gps:
            gc = GPS_COLS[robot.gps_mode]
            pygame.draw.circle(screen, gc,         (int(gps[0]), int(gps[1])), GPS_DOT_RAD)
            pygame.draw.circle(screen, (255,255,255),(int(gps[0]), int(gps[1])), GPS_DOT_RAD, 1)

        # Odometry estimate dot
        pygame.draw.circle(screen, (90,140,255),
                           (int(robot.odo_x), int(robot.odo_y)), 4)

        # Robot body + arrow
        pygame.draw.circle(screen, ROBOT_COL, (int(robot.x), int(robot.y)), robot.radius)
        pygame.draw.circle(screen, (255,255,255),(int(robot.x), int(robot.y)), robot.radius, 2)
        draw_arrow(screen, (255,255,255), int(robot.x), int(robot.y), robot.heading)

        # ── HUD ────────────────────────────────────────────────────────
        hud_surf.fill((22, 22, 30, 230))

        # GPS mode badge
        gm   = robot.gps_mode.upper()
        gcol = GPS_COLS[robot.gps_mode]
        badge = font_big.render(f"GPS: {gm}", True, gcol)
        hud_surf.blit(badge, (12, 10))
        hint = font_small.render("Press G to cycle", True, (90,90,110))
        hud_surf.blit(hint, (12, 32))

        # Odometry drift
        drift = math.hypot(robot.x - robot.odo_x, robot.y - robot.odo_y)
        dr_col = (52,199,142) if drift < 10 else (255,160,40) if drift < 30 else (230,80,80)
        dr_lbl = font_big.render(f"Odo drift: {drift:.1f} px", True, dr_col)
        hud_surf.blit(dr_lbl, (220, 10))

        # Distance to goal
        dg_lbl = font_big.render(f"Dist to goal: {dist_to_goal:.0f} px", True, (180,180,200))
        hud_surf.blit(dg_lbl, (220, 32))

        # Steps + time
        elapsed = time.time() - start_time
        st_lbl = font_small.render(f"Steps: {steps}   Time: {elapsed:.1f}s   Speed: {robot.speed:.2f}", True, (90,90,110))
        hud_surf.blit(st_lbl, (12, 56))

        # Legend
        legends = [
            ((52,199,142), "True pos / trail"),
            ((90,140,255), "Odometry estimate"),
            (GPS_COLS[robot.gps_mode], "GPS reading"),
        ]
        for i, (lc, lt) in enumerate(legends):
            pygame.draw.circle(hud_surf, lc, (440 + 0, 14 + i*22), 5)
            ll = font_small.render(lt, True, (160,160,180))
            hud_surf.blit(ll, (452, 8 + i*22))

        # Controls reminder
        ctrl = font_small.render("WASD / Arrows = move    G = GPS mode    R = reset    ESC = quit", True, (60,60,80))
        hud_surf.blit(ctrl, (12, 78))

        screen.blit(hud_surf, (0, H))
        pygame.display.flip()

if __name__ == "__main__":
    main()