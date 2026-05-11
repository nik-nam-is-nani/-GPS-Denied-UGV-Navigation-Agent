# GPS-Denied UGV Navigation Agent

A reinforcement learning-based navigation system for unmanned ground vehicles operating in GPS-denied environments. Built for DRDO's MUNTRA program requirements.

## Features

- **GPS Spoofing Simulation**: Simulates jamming, spoofing, and drift attacks
- **Visual Odometry**: ORB-feature based motion estimation with drift
- **EKF Sensor Fusion**: Extended Kalman Filter combining VO + IMU
- **Procedural Terrain**: Multiple difficulty levels (easy, medium, hard, featureless)
- **PPO Training**: Curriculum learning from simple to complex scenarios
- **Real-time Dashboard**: WebSocket-based live visualization

## Project Structure

```
ugv-gps-denied/
├── env/                    # Gym environment
│   ├── ugv_env.py         # Main RL environment
│   ├── terrain.py         # Map generation
│   └── gps_spoofer.py     # GPS attack simulator
├── perception/            # Sensor processing
│   ├── visual_odometry.py # Feature tracking
│   ├── depth_estimator.py # Depth estimation
│   ├── imu_fusion.py      # EKF filter
│   └── occupancy_map.py   # 2D grid mapping
├── agent/                 # RL training
│   ├── train.py           # Training script
│   ├── reward.py          # Reward function
│   ├── policy.py          # Custom policy
│   └── metrics.py         # ATE/RPE evaluation
├── api/
│   └── server.py          # FastAPI + WebSocket server
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent (Optional - takes ~30 min on GPU)

```bash
cd agent
python train.py
# Choose option 1 for quick test, 2 for full curriculum
```

### 3. Run the Server

```bash
cd api
python server.py
```

### 4. View Dashboard

Open browser at: `http://localhost:8000`

## Running Modes

### Training
```bash
cd agent
python train.py
# Enter: 1 for quick test (10K steps)
#     : 2 for full curriculum (2M steps)
#     : 3 for single stage
```

### Evaluation Only (No Training)
```bash
cd api
python server.py
```

The server will run with random actions if no trained model exists.

### Test Environment Manually
```bash
python -c "
from env.ugv_env import UGVNavEnv
env = UGVNavEnv(map_size=100, gps_denied=True)
obs, _ = env.reset()
for i in range(50):
    action = env.action_space.sample()
    obs, r, t, tr, info = env.step(action)
    if t or tr: break
print('Test complete!')
"
```

## GPS Attack Modes

| Mode | Description | Use |
|------|-------------|-----|
| `none` | Normal GPS | Baseline |
| `jam` | Signal blocked | No position fix |
| `spoof` | False coordinates | Off by 10-50m |
| `drift` | Slowly wrong | Gradual drift |

Set via: `gps_spoofer.set_mode("jam")`

## Metrics Tracked

- **ATE**: Absolute Trajectory Error (mean position error)
- **RPE**: Relative Pose Error (drift rate)
- **Success Rate**: Goal reach percentage
- **Collision Rate**: Obstacle hit percentage
- **VO Confidence**: Visual odometry reliability

## Technical Details

- **Observation Space**: 15 dims (position, heading, goal direction, VO confidence, 8 lidar rays)
- **Action Space**: [steering: -1 to 1, throttle: 0 to 1]
- **Max Steps**: 500 per episode
- **Map Size**: 200x200 (configurable)

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- Stable-Baselines3
- FastAPI
- OpenCV

## License

MIT