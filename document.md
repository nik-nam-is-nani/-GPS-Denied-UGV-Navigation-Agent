# UGV GPS-Denied Navigation Simulator — Phase 1

A 2D vehicle simulation built with Pygame as the foundation
for a DRDO-relevant GPS-denied UGV navigation project.

## Run it
```bash
pip install -r requirements.txt
python main.py
```

## Controls
| Key | Action |
|-----|--------|
| W / Up    | Accelerate forward |
| S / Down  | Brake / reverse |
| A / Left  | Steer left |
| D / Right | Steer right |
| G         | Cycle GPS mode (on → off → spoof → drift) |
| R         | Reset robot |
| ESC       | Quit |

## What you can already see
- Green dot + arrow = true robot position
- Blue dot = odometry (dead-reckoning) estimate — drifts from true over time
- Coloured dot = GPS reading (changes with mode)
- Green rays = rangefinder / sensor rays hitting obstacles
- Trails = history of true path (green) vs odometry estimate (blue)

## GPS Modes
| Mode  | What happens |
|-------|--------------|
| ON    | Clean GPS with small noise |
| OFF   | No GPS — only odometry |
| SPOOF | GPS reports wrong location (attack) |
| DRIFT | GPS slowly walks away from truth |

## Phase Roadmap
- [x] Phase 1 — Pygame simulation with physics + sensors + GPS modes
- [ ] Phase 2 — Add A* pathfinding (auto-navigate to goal)
- [ ] Phase 3 — Convert to Gymnasium environment
- [ ] Phase 4 — Train PPO agent with Stable-Baselines3
- [ ] Phase 5 — FastAPI + React dashboard
- [ ] Phase 6 — HuggingFace deployment