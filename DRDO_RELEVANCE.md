# DRDO CAIR Relevance

## Project Connection to CAIR Research Programs

This project directly addresses requirements from DRDO's Centre for Artificial Intelligence Research (CAIR) and aligns with operational needs of India's unmanned systems.

---

## MUNTRA-S (Manned-Unmanned Teaming)

**Relevant Components:**
- GPS-denied navigation for tactical environments
- Visual odometry as fallback positioning
- Curriculum learning for terrain adaptation

**How We Address It:**
Our EKF-based sensor fusion provides the precise positioning needed for MUNTRA's convoy operations. The VO drift penalty in our reward function ensures the agent learns to operate in visually challenging environments (tunnels, urban canyons) where GPS fails.

---

## MARF (Multi-Agent Robotics Framework)

**Relevant Components:**
- Distributed sensing and localization
- Inter-agent position estimation

**How We Address It:**
The occupancy mapping module can be extended to share map data between multiple UGVs, creating a collaborative SLAM system essential for MARF's swarm operations.

---

## ATR (Autonomous Target Recognition)

**Relevant Components:**
- Real-time perception in contested environments

**How We Address It:**
Our depth estimation pipeline (MiDaS-based) provides terrain classification that can be fused with target detection systems for battlefield awareness.

---

## Electronic Warfare Relevance

### GPS Spoofing Countermeasures

This project explicitly models three attack vectors:

1. **Jamming**: Complete GPS signal loss - tests navigation resilience
2. **Spoofing**: False position injection - tests anomaly detection
3. **Drift**: Slow position corruption - tests VO quality recovery

This is directly relevant to India's electronic warfare doctrine as outlined in strategic documents on counter-GPS navigation.

---

## Technical Capabilities for DRDO

| Capability | CAIR Program Alignment | Project Module |
|------------|----------------------|----------------|
| Sensor fusion (VO + IMU) | MUNTRA, MARF | `imu_fusion.py` |
| Real-time mapping | MARF, ATR | `occupancy_map.py` |
| GPS attack simulation | Electronic Warfare | `gps_spoofer.py` |
| Curriculum learning | All programs | `train.py` |
| ATE/RPE metrics | MUNTRA validation | `metrics.py` |

---

## Future Extensions

1. **LiDAR Integration**: Add point-cloud processing for 3D mapping
2. **Edge Deployment**: Quantize model for embedded hardware
3. **Multi-UGV SLAM**: Extend occupancy map for distributed operation
4. **DRDO Dataset Integration**: Fine-tune on CAIR's field data

---

*Project developed for educational/research purposes aligned with DRDO-CAIR objectives.*