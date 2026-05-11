"""
UGV GPS-Denied Navigation Simulator — GPS Attack Proof-of-Concept
Demonstrates A* navigation failure under GPS spoofing, jamming & drift.

Controls:
  W/UP, S/DOWN, A/LEFT, D/RIGHT — manual control (when not auto-piloting)
  G — cycle GPS mode: NORMAL → JAMMED → SPOOFED → DRIFT
  R — reset simulation
  CLICK anywhere on map — set goal & start auto-pilot

SPOOFED: A* replans from fake GPS position → robot navigates wrong
JAMMED: No GPS → path cannot replan → robot may get lost
"""

import pygame, sys, math, time, random
from map    import GridMap
from robot  import Robot
from astar  import astar

# ── Colours ──────────────────────────────────────────────────────────────────
BG          = (10,  12,  16)
WALL_COL    = (45,  50,  60)
EMPTY_COL   = (20,  24,  30)
ROBOT_COL   = ( 0, 170, 255)   # BLUE = true position
GPS_COL     = (255,  80,  80)   # RED  = GPS-reported (spoofed/fake)
GOAL_COL    = ( 60, 255, 120)  # GREEN goal marker
PATH_COL    = (255, 200,  60)  # YELLOW A* path
DIVERGE_COL = (255,  40,  40, 180)  # dashed red diversion line
HUD_BG      = ( 8,  10,  14, 240)
HUD_FG      = (160, 200, 255)
WARN_COL    = (255,   0,   0)
WARN_BLINK  = (255,  60,  60)

FONT_BIG   = None
FONT_SMALL = None

# ── GPS mode definitions ─────────────────────────────────────────────────────
GPS_MODES = ["normal", "jammed", "spoofed", "drift"]

# ── Navigation confidence per mode ───────────────────────────────────────────
CONFIDENCE = {
    "normal":  100,
    "drift":   70,
    "spoofed": 25,
    "jammed":   0,
}

# ── Spoof offsets (world pixels) ──────────────────────────────────────────────
SPOOF_OFFSET = (120, -80)

# ── Replan interval (seconds) ─────────────────────────────────────────────────
REPLAN_INTERVAL = 2.0

# ── Screen shake ──────────────────────────────────────────────────────────────
SHAKE_INTENSITY = {
    "normal":   0,
    "drift":    2,
    "spoofed":  5,
    "jammed":   3,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_arrow(surf, colour, x, y, angle, length=16, width=3):
    ex = x + length * math.cos(angle)
    ey = y + length * math.sin(angle)
    pygame.draw.line(surf, colour, (x, y), (ex, ey), width)
    for side in (-0.5, 0.5):
        ax = ex + 8 * math.cos(angle + math.pi + side)
        ay = ey + 8 * math.sin(angle + math.pi + side)
        pygame.draw.line(surf, colour, (ex, ey), (ax, ay), width)

def draw_dashed(surf, colour, pts, dash=6, gap=4, width=2):
    """Draw a dashed line connecting points."""
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        x0, y0 = int(pts[i-1][0]), int(pts[i-1][1])
        x1, y1 = int(pts[i][0]),   int(pts[i][1])
        dx, dy = x1 - x0, y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        nx, ny = dx / seg_len, dy / seg_len
        pos = 0.0
        drawing = True
        while pos < seg_len:
            end = min(pos + (dash if drawing else gap), seg_len)
            if drawing:
                px = x0 + nx * pos
                py = y0 + ny * pos
                qx = x0 + nx * end
                qy = y0 + ny * end
                pygame.draw.line(surf, colour, (int(px), int(py)), (int(qx), int(qy)), width)
            pos = end
            drawing = not drawing

def world_to_tile(grid, x, y):
    """Convert world pixels → tile (col, row)."""
    return int(x // grid.tile), int(y // grid.tile)

def tile_to_world(grid, col, row):
    """Return world center of tile."""
    return col * grid.tile + grid.tile // 2, row * grid.tile + grid.tile // 2

def compute_gps_position(robot, mode):
    """Return GPS-reported (x, y) based on current mode."""
    if mode == "normal":
        # Tiny noise to mimic imperfect GPS
        nx = robot.true_x + (random.random() * 1.6 - 0.8)
        ny = robot.true_y + (random.random() * 1.6 - 0.8)
        return nx, ny
    elif mode == "spoofed":
        # Fixed spoof offset from robot's true position
        return robot.true_x + SPOOF_OFFSET[0], robot.true_y + SPOOF_OFFSET[1]
    elif mode == "jammed":
        return None  # no fix
    elif mode == "drift":
        # Accumulate slowly-growing drift
        robot._drift_accum[0] += (random.random() * 0.8 - 0.4)
        robot._drift_accum[1] += (random.random() * 0.8 - 0.4)
        return robot.true_x + robot._drift_accum[0], robot.true_y + robot._drift_accum[1]
    return robot.true_x, robot.true_y

def compute_confidence(mode):
    return CONFIDENCE.get(mode, 0)

# ═══════════════════════════════════════════════════════════════════════════════
# Path planning wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def plan_path(grid, start_pos, goal_tile):
    """
    Compute A* path from a starting world position to a goal tile.
    start_pos: (x, y) world pixels
    goal_tile: (col, row) tile coordinates
    Returns: list of world (x, y) waypoints, or [] if unreachable.
    """
    start_tile = world_to_tile(grid, start_pos[0], start_pos[1])

    # Path cannot start inside a wall — abort planning
    if grid.is_wall(start_tile[0], start_tile[1]):
        return []

    path_tiles = astar(grid, start_tile, goal_tile)
    if not path_tiles:
        return []

    # Convert tile centers → world coordinates
    return [tile_to_world(grid, c, r) for c, r in path_tiles]

# ═══════════════════════════════════════════════════════════════════════════════
# Event log
# ═══════════════════════════════════════════════════════════════════════════════

class EventLog:
    """Scrolling log of timestamped events."""
    def __init__(self, max_entries=8):
        self.entries = []
        self.max = max_entries

    def log(self, msg):
        t = time.time() - self.start_time
        ts = f"[{t:4.1f}s]"
        self.entries.insert(0, f"{ts} {msg}")
        if len(self.entries) > self.max:
            self.entries.pop()

    def draw(self, surf, font, x, y, col=(180, 220, 255)):
        for i, entry in enumerate(self.entries):
            lbl = font.render(entry, True, col)
            surf.blit(lbl, (x, y + i * 16))

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    global FONT_BIG, FONT_SMALL
    pygame.init()
    FONT_BIG   = pygame.font.SysFont("consolas", 16, bold=True)
    FONT_SMALL = pygame.font.SysFont("consolas", 13)

    grid = GridMap(cols=40, rows=30, tile=20)
    W, H = grid.width, grid.height
    HUD_H = 140
    screen = pygame.display.set_mode((W, H + HUD_H))
    pygame.display.set_caption("UGV GPS-SPOOFING / JAMMING DEFENSE POC")
    clock  = pygame.time.Clock()

    # ── Surfaces ───────────────────────────────────────────────────────────────
    map_surf   = pygame.Surface((W, H))
    hud_surf   = pygame.Surface((W, HUD_H), pygame.SRCALPHA)
    path_surf  = pygame.Surface((W, H),     pygame.SRCALPHA)  # A* path (yellow)
    divert_surf= pygame.Surface((W, H),     pygame.SRCALPHA)  # GPS offset line (red dashed)
    true_surf  = pygame.Surface((W, H),     pygame.SRCALPHA)  # true trail (cyan)
    gps_surf   = pygame.Surface((W, H),     pygame.SRCALPHA)  # GPS-reported markers

    # Pre-draw static map
    for row in range(grid.rows):
        for col in range(grid.cols):
            rect = pygame.Rect(col*grid.tile, row*grid.tile, grid.tile, grid.tile)
            col_c = WALL_COL if grid.grid[row, col] else EMPTY_COL
            pygame.draw.rect(map_surf, col_c, rect)
    # Grid lines (subtle)
    for r in range(grid.rows+1):
        pygame.draw.line(map_surf, (30,35,42), (0, r*grid.tile), (W, r*grid.tile))
    for c in range(grid.cols+1):
        pygame.draw.line(map_surf, (30,35,42), (c*grid.tile, 0), (c*grid.tile, H))

    # Robot + fixed goal
    start_px, start_py = tile_to_world(grid, 3, 3)
    goal_tile   = (35, 25)              # fixed goal tile
    goal_px, goal_py = tile_to_world(grid, *goal_tile)
    robot = Robot(start_px, start_py, heading=0.0)

    # ── Simulation state ───────────────────────────────────────────────────────
    gps_mode     = "normal"
    auto_pilot   = False
    path         = []    # list of (x, y) world waypoints
    current_wp   = 0     # index in path
    last_replan  = 0.0   # time of last A* recompute
    spoof_alerted = False  # tracks if we've already logged spoof divergence
    event_log    = EventLog(max_entries=10)
    event_log.start_time = time.time()
    event_log.log("SIMULATION STARTED — CLICK MAP TO SET GOAL")

    blink_state = False
    running = True
    while running:
        dt       = clock.tick(60)                       # cap 60 FPS
        now      = time.time()
        shake    = int(SHAKE_INTENSITY[gps_mode] * math.sin(now * 20))

        # ── Events ─────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_g:
                    # Cycle GPS mode
                    idx = GPS_MODES.index(gps_mode)
                    gps_mode = GPS_MODES[(idx + 1) % len(GPS_MODES)]
                    # Reset drift accumulator on mode changes
                    robot._drift_accum[:] = 0
                    robot._spoof_offset = list(SPOOF_OFFSET)
                    spoof_alerted = False  # reset spoof warning flag
                    event_log.log(f"GPS MODE → {gps_mode.upper()}")
                    # On mode switch, force an immediate replan
                    last_replan = 0.0
                if ev.key == pygame.K_r:
                    # Full reset
                    robot = Robot(start_px, start_py, heading=0.0)
                    robot._spoof_offset = list(SPOOF_OFFSET)
                    auto_pilot   = False
                    path         = []
                    current_wp   = 0
                    spoof_alerted = False
                    gps_pos      = compute_gps_position(robot, gps_mode)
                    path_surf.fill((0,0,0,0))
                    divert_surf.fill((0,0,0,0))
                    true_surf.fill((0,0,0,0))
                    gps_surf.fill((0,0,0,0))
                    last_replan  = 0.0
                    event_log    = EventLog(max_entries=10)
                    event_log.start_time = time.time()
                    event_log.log("SIMULATION RESET")
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if my < H:
                    # Click sets new goal — plan path from current GPS position
                    goal_tile_px = (mx // grid.tile, my // grid.tile)
                    if not grid.is_wall(*goal_tile_px):
                        if gps_pos is not None:
                            new_path = plan_path(grid, gps_pos, goal_tile_px)
                            if new_path:
                                path          = new_path
                                current_wp    = 0
                                auto_pilot    = True
                                last_replan   = now
                                event_log.log(f"GOAL SET — PATH PLANNED ({len(path)} waypoints)")
                        else:
                            event_log.log("CANNOT PLAN — GPS JAMMED (NO FIX)")

        # ── Compute current GPS fix ─────────────────────────────────────────────
        gps_pos  = compute_gps_position(robot, gps_mode)

        # ── Dynamic replanning ───────────────────────────────────────────────────
        needs_replan = False
        if auto_pilot and path and current_wp < len(path):
            # Replan every REPLAN_INTERVAL seconds from the latest GPS position
            if now - last_replan >= REPLAN_INTERVAL:
                if gps_pos is not None:
                    new_path = plan_path(grid, gps_pos, goal_tile)
                    if new_path and new_path != path:
                        path        = new_path
                        current_wp  = 0  # restart from closest new waypoint
                        last_replan = now
                        event_log.log("PATH RECALCULATED (GPS UPDATE)")
                        needs_replan = True
                        spoof_alerted = False  # allow fresh spoof alert for new path
                else:
                    # Jammed — cannot replan
                    pass

        # ── Control: Auto-pilot follows current waypoint ────────────────────────
        throttle = 0.0
        steer    = 0.0

        if auto_pilot and path and current_wp < len(path):
            wp_x, wp_y = path[current_wp]
            dx = wp_x - robot.x
            dy = wp_y - robot.y
            dist = math.hypot(dx, dy)

            if dist < 10:                      # waypoint reached
                current_wp += 1
                if current_wp >= len(path):
                    auto_pilot = False
                    event_log.log("GOAL REACHED!")
            else:
                desired = math.atan2(dy, dx)
                err = (desired - robot.heading + math.pi) % (2*math.pi) - math.pi
                steer = max(-1.0, min(1.0, err * 3.0))
                throttle = 1.0 if abs(err) < 0.6 else 0.5

        # ── Robot update ─────────────────────────────────────────────────────────
        robot.update(throttle, steer, grid)

        # ── Navigation error (distance between true pos and GPS-reported pos) ────
        if gps_pos:
            nav_error = math.hypot(robot.true_x - gps_pos[0], robot.true_y - gps_pos[1])
        else:
            nav_error = float('inf')

        # Collision detection (ray-based proximity warning)
        rays = robot.cast_rays(grid, num_rays=6, max_dist=80)
        collision_risk = any(dist < 12 for _, _, dist in rays)
        if collision_risk and not auto_pilot:
            event_log.log("COLLISION WARNING — OBSTACLE NEAR")

        # ── Warn on spoofing path deviation ─────────────────────────────────────
        if auto_pilot and gps_mode == "spoofed" and not needs_replan:
            # Compare current A* path with true-position ideal path
            true_start = (robot.true_x, robot.true_y)
            ideal_path = plan_path(grid, true_start, goal_tile)
            if ideal_path:
                # Measure first major divergence
                min_len = min(len(path), len(ideal_path))
                diverges = any(
                    math.hypot(path[i][0]-ideal_path[i][0], path[i][1]-ideal_path[i][1]) > 40
                    for i in range(min_len)
                )
                if diverges and not spoof_alerted:
                    event_log.log(f"NAV ERROR: {nav_error:.0f}px — PATH CORRUPTED")
                    spoof_alerted = True

        # ── Draw ─────────────────────────────────────────────────────────────────
        screen.fill(BG)
        screen.blit(map_surf, (0, 0))

        # Screen shake offset
        ox, oy = shake, shake

        # Clear transient overlays each frame
        true_surf.fill((0,0,0,0))
        gps_surf.fill((0,0,0,0))

        # True trail (cyan)
        if len(robot.true_trail) >= 2:
            pts = [(int(p[0])+ox, int(p[1])+oy) for p in robot.true_trail[-120:]]
            pygame.draw.lines(true_surf, (0, 200, 255, 70), False, pts, 2)
            screen.blit(true_surf, (0, 0))

        # A* planned path (yellow) — drawn from GPS-replanning perspective
        if path and current_wp < len(path):
            path_surf.fill((0,0,0,0))
            pts = [(int(p[0])+ox, int(p[1])+oy) for p in path]
            if len(pts) >= 2:
                pygame.draw.lines(path_surf, PATH_COL, False, pts, 3)
            # Current waypoint indicator
            wx, wy = pts[current_wp]
            pygame.draw.circle(path_surf, (255,255,255), (wx, wy), 6, 2)
            screen.blit(path_surf, (0, 0))

        # Red dashed line: true position → GPS-reported position
        if gps_pos and gps_mode != "normal":
            divert_surf.fill((0,0,0,0))
            true_px = (int(robot.true_x), int(robot.true_y))
            gps_px  = (int(gps_pos[0]),       int(gps_pos[1]))
            draw_dashed(divert_surf, DIVERGE_COL, [true_px, gps_px], dash=5, gap=4, width=2)
            screen.blit(divert_surf, (0, 0))

        # GPS-reported marker (red circle with outline) — spoofed / drifted position
        if gps_pos:
            gx, gy = int(gps_pos[0])+ox, int(gps_pos[1])+oy
            pygame.draw.circle(gps_surf, (255, 80, 80, 200), (gx, gy), 8)
            pygame.draw.circle(gps_surf, (255,255,255),        (gx, gy), 8, 2)
            screen.blit(gps_surf, (0, 0))

        # Goal marker
        pygame.draw.circle(screen, GOAL_COL, (goal_px, goal_py), 10)
        pygame.draw.circle(screen, (255,255,255), (goal_px, goal_py), 10, 2)

        # The robot itself (blue circle + arrow) — TRUE position
        robot.draw(screen, ox, oy)

        # ── HUD Panel ────────────────────────────────────────────────────────────
        hud_surf.fill(HUD_BG)

        confidence = compute_confidence(gps_mode)
        conf_col = (52, 255, 120) if confidence >= 80 else (255,200,60) if confidence >= 40 else (255,80,80)

        # Left column — status telemetry
        x_left = 10
        y = 8
        lbl = FONT_SMALL.render(f"GPS:   {gps_mode.upper()}", True, (255,80,80) if gps_mode=="spoofed" else (255,200,60) if gps_mode=="jammed" else (52,255,120))
        hud_surf.blit(lbl, (x_left, y)); y += 20

        lbl = FONT_SMALL.render(f"TRUE:  ({robot.true_x:5.0f}, {robot.true_y:5.0f})", True, (0,200,255))
        hud_surf.blit(lbl, (x_left, y)); y += 20

        if gps_pos:
            lbl = FONT_SMALL.render(f"GPS:   ({gps_pos[0]:5.0f}, {gps_pos[1]:5.0f})", True, (255,80,80))
        else:
            lbl = FONT_SMALL.render("GPS:   NO FIX — JAMMED", True, (255,80,80))
        hud_surf.blit(lbl, (x_left, y)); y += 20

        err_text = f"{nav_error:.0f} px" if not math.isinf(nav_error) else "--- px"
        err_col  = (255,200,60) if not math.isinf(nav_error) and nav_error > 50 else (160,200,255)
        lbl = FONT_SMALL.render(f"NAV ERROR: {err_text}", True, err_col)
        hud_surf.blit(lbl, (x_left, y)); y += 20

        lbl = FONT_SMALL.render(f"TARGET WP: {current_wp+1}/{len(path) if path else 0}", True, HUD_FG)
        hud_surf.blit(lbl, (x_left, y)); y += 20

        lbl = FONT_SMALL.render(f"REPLAN:  {'ACTIVE' if auto_pilot else 'IDLE'}  every {REPLAN_INTERVAL:.0f}s", True,
                                (52,255,120) if auto_pilot else (120,120,140))
        hud_surf.blit(lbl, (x_left, y)); y += 20

        # Center — confidence meter
        conf_col = (52,255,120) if confidence >= 80 else (255,200,60) if confidence >= 40 else (255,80,80)
        lbl = FONT_BIG.render(f"CONFIDENCE: {confidence}%", True, conf_col)
        hud_surf.blit(lbl, (W//2 - lbl.get_width()//2, 8))

        # Right column — event log (scrolling)
        event_log.draw(hud_surf, FONT_SMALL, 280, 8)

        screen.blit(hud_surf, (0, H))

        # ── Warning Banner ───────────────────────────────────────────────────────
        if gps_mode in ("spoofed", "jammed"):
            banner_h = 28
            color = WARN_BLINK if blink_state else (40, 0, 0)
            pygame.draw.rect(screen, color, (0, 0, W, banner_h))
            pygame.draw.line(screen, WARN_COL, (0, banner_h), (W, banner_h), 2)
            if gps_mode == "spoofed":
                txt1 = "⚠  GPS SPOOFING DETECTED — NAVIGATION COMPROMISED  ⚠"
                txt2 = "A* replanning from FAKE position — expect incorrect path"
            else:  # jammed
                txt1 = "⚠  GPS SIGNAL LOST — NO POSITION FIX  ⚠"
                txt2 = "Replanning disabled — robot relying on odometry / last known state"
            l1 = FONT_BIG.render(txt1, True, WARN_COL)
            l2 = FONT_SMALL.render(txt2, True, (255, 120, 120))
            screen.blit(l1, (W//2 - l1.get_width()//2, 4))
            screen.blit(l2, (W//2 - l2.get_width()//2, banner_h - 18))
            # Blink every half-second
            blink_state = not blink_state
            # Add screen shake for dramatic effect
            if shake:
                screen.scroll(shake - shake//2, 0)

        pygame.display.flip()

if __name__ == "__main__":
    main()
