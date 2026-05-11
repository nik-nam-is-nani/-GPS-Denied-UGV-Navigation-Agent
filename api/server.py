from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import asyncio
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Importing stable_baselines3...")
from stable_baselines3 import PPO
print("Importing env...")
from env.ugv_env import UGVNavEnv
print("Imports complete")


app = FastAPI(title="UGV GPS-Denied Navigation")

model = None
env = None
simulation_active = False


@app.on_event("startup")
async def load_model():
    global model, env
    model_path = "./models/ugv_final.zip"

    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("Loaded trained model")
    else:
        print("No trained model found, using random actions")

    env = UGVNavEnv(map_size=200, max_steps=500, gps_denied=True, difficulty="medium")


@app.get("/")
async def get_index():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>UGV GPS-Denied Navigation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        h1 { color: #00d4ff; margin-bottom: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            backdrop-filter: blur(10px);
        }
        #map { border: 2px solid #00d4ff; border-radius: 5px; }
        .metrics { width: 300px; }
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric-value { color: #00d4ff; font-weight: bold; }
        .controls { margin-top: 15px; }
        button {
            background: #00d4ff;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 10px;
        }
        button:hover { background: #00a8cc; }
        select {
            padding: 10px;
            border-radius: 5px;
            background: #2a2a4a;
            color: #fff;
            border: 1px solid #00d4ff;
        }
        .status { padding: 10px; border-radius: 5px; margin-top: 10px; }
        .status.connected { background: #00ff8833; }
        .status.disconnected { background: #ff444433; }
    </style>
</head>
<body>
    <h1>🚜 UGV GPS-Denied Navigation Simulator</h1>
    <div class="controls">
        <button onclick="startSimulation()">Start</button>
        <button onclick="stopSimulation()">Stop</button>
        <select id="gpsMode">
            <option value="none">GPS Normal</option>
            <option value="jam">GPS Jammed</option>
            <option value="spoof">GPS Spoofed</option>
            <option value="drift">GPS Drifting</option>
        </select>
    </div>
    <div id="status" class="status disconnected">Disconnected</div>
    <br>
    <div class="container">
        <div class="panel">
            <canvas id="map" width="500" height="500"></canvas>
        </div>
        <div class="panel metrics">
            <h3>Real-time Metrics</h3>
            <div class="metric-row">
                <span>Step</span>
                <span class="metric-value" id="step">0</span>
            </div>
            <div class="metric-row">
                <span>True Position</span>
                <span class="metric-value" id="truePos">0, 0</span>
            </div>
            <div class="metric-row">
                <span>Est. Position</span>
                <span class="metric-value" id="estPos">0, 0</span>
            </div>
            <div class="metric-row">
                <span>VO Drift</span>
                <span class="metric-value" id="drift">0.00</span>
            </div>
            <div class="metric-row">
                <span>Reward</span>
                <span class="metric-value" id="reward">0.00</span>
            </div>
            <div class="metric-row">
                <span>GPS Status</span>
                <span class="metric-value" id="gpsStatus">OFF</span>
            </div>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('map');
        const ctx = canvas.getContext('2d');
        let ws = null;
        let running = false;
        let mapData = null;
        let truePos = [0, 0];
        let estPos = [0, 0];
        let goal = [0, 0];

        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws/simulate');
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
            };
            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'status disconnected';
                running = false;
            };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                updateDisplay(data);
            };
        }

        function startSimulation() {
            if (!ws || ws.readyState !== WebSocket.OPEN) connect();
            running = true;
            setTimeout(() => ws.send('start'), 100);
        }

        function stopSimulation() {
            running = false;
            if (ws) ws.send('stop');
        }

        function updateDisplay(data) {
            truePos = data.true_pos;
            estPos = data.est_pos;
            goal = data.goal;

            document.getElementById('step').textContent = data.step;
            document.getElementById('truePos').textContent =
                truePos[0].toFixed(1) + ', ' + truePos[1].toFixed(1);
            document.getElementById('estPos').textContent =
                estPos[0].toFixed(1) + ', ' + estPos[1].toFixed(1);
            document.getElementById('drift').textContent = data.drift.toFixed(2);
            document.getElementById('reward').textContent = data.reward.toFixed(2);

            drawMap(data.map);
        }

        function drawMap(grid) {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, 500, 500);

            if (grid && grid.length > 0) {
                const cellW = 500 / grid[0].length;
                const cellH = 500 / grid.length;

                for (let y = 0; y < grid.length; y++) {
                    for (let x = 0; x < grid[0].length; x++) {
                        if (grid[y][x] === 1) {
                            ctx.fillStyle = '#444';
                            ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
                        }
                    }
                }
            }

            ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
            ctx.beginPath();
            ctx.arc(goal[0] * 2.5, goal[1] * 2.5, 8, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            ctx.beginPath();
            ctx.arc(truePos[0] * 2.5, truePos[1] * 2.5, 5, 0, Math.PI * 2);
            ctx.fill();

            ctx.strokeStyle = 'rgba(0, 212, 255, 0.6)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(estPos[0] * 2.5, estPos[1] * 2.5, 5, 0, Math.PI * 2);
            ctx.stroke();
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(html)


@app.websocket("/ws/simulate")
async def simulate(websocket: WebSocket):
    await websocket.accept()
    global simulation_active

    try:
        simulation_active = True
        obs, _ = env.reset()

        while simulation_active:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            response = {
                "true_pos": info['true_pos'].tolist(),
                "est_pos": info['estimated_pos'].tolist(),
                "goal": env.goal.tolist(),
                "reward": float(reward),
                "step": env.step_count,
                "drift": float(info['drift']),
                "map": env.grid.tolist() if hasattr(env, 'grid') else []
            }

            await websocket.send_json(response)

            if terminated or truncated:
                obs, _ = env.reset()

            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        simulation_active = False
    except Exception as e:
        print(f"WebSocket error: {e}")
        simulation_active = False


@app.get("/api/status")
async def get_status():
    return {
        "model_loaded": model is not None,
        "simulation_active": simulation_active
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)