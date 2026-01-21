# KRTI 2024 - Team SIRI (Soegijapranata Catholic University)

**Division:** KRTI 2024 (Kontes Robot Terbang Indonesia)
**Team:** SIRI
**Vehicle Type:** Hexarotor (VTOL)

This repository contains the complete autonomous flight control and computer vision software for Team SIRI's hexarotor UAV, representing Soegijapranata Catholic University (SCU) in the 2024 Indonesian Aerial Robotic Contest (KRTI).

## 🚁 System Overview

The system is designed to perform autonomous navigation and mission-specific tasks (such as object detection and payload delivery) using a companion computer setup.

### Hardware Architecture
* **Airframe:** Custom Hexarotor Frame
* **Flight Controller (Slave):** Pixhawk (handling attitude, motor mixing, and failsafes)
* **Onboard Computer (Master):** Raspberry Pi 4 (handling high-level mission logic, computer vision, and communication)
* **Communication:** UART/Serial connection between Raspberry Pi and Pixhawk (MAVLink protocol)

### Software Features
* **Autonomous Flight:** Waypoint navigation, auto-takeoff, and auto-landing logic.
* **Computer Vision (AI):** Real-time object detection using **PyTorch** models (`.pt` files included) to identify mission targets.
* **Live Monitoring:** Video streaming capability via a web interface.
* **Dual-Environment Modes:** Specific logic for both indoor testing and outdoor GPS-based missions.

## 📂 Repository Structure

* `krti_full_program.py`: Main execution script for the autonomous mission.
* `krti_full_program_ver2.py`: Updated version of the main flight logic.
* `krti_outdoor_program.py`: Specialized logic for outdoor missions involving GPS coordinates.
* `stream_video_webpage.html`: Web interface for monitoring the onboard camera feed live.
* `best_batch10_epochs150.pt` / `best_batch20_epochs150.pt`: Pre-trained PyTorch model weights for object detection.

## 🛠️ Technology Stack

* **Language:** Python 3.x
* **Flight Control:** DroneKit / Pymavlink
* **Computer Vision:** PyTorch (Ultralytics YOLO inferred), OpenCV
* **Web Streaming:** HTML5, HTTP Server

## ⚙️ Installation & Setup

These instructions assume you are setting up on a Raspberry Pi 4 running Raspberry Pi OS (Legacy/Buster recommended for camera compatibility).

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/felixyustian/krti2024_scu_unika.git](https://github.com/felixyustian/krti2024_scu_unika.git)
    cd krti2024_scu_unika
    ```

2.  **Install Dependencies**
    *(Ensure you are in a virtual environment if preferred)*
    ```bash
    # Install PyTorch (ARM64 version for Raspberry Pi)
    pip3 install torch torchvision

    # Install DroneKit for MAVLink communication
    pip3 install dronekit

    # Install OpenCV for image processing
    pip3 install opencv-python

    # Install other utilities
    pip3 install pyserial numpy
    ```

3.  **Hardware Connection**
    * Connect the Pixhawk `TELEM 2` port to the Raspberry Pi `GPIO 14 (TX)` and `GPIO 15 (RX)`.
    * Ensure the camera is connected and enabled (`raspi-config`).

## 🚀 Usage

### 1. Pre-Flight Checks
Ensure the drone acts as a WiFi hotspot or is connected to the same network as your Ground Control Station (GCS).

### 2. Running the Autonomous Mission
Execute the main program script. Ensure the Pixhawk is armed and in `GUIDED` mode if not handled automatically by the script.

```bash
# For standard mission logic
python3 krti_full_program_ver2.py

# For outdoor GPS-based missions
python3 krti_outdoor_program.py
