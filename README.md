# 👔 Smart Clothing Store Autonomous Serving System

![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?style=for-the-badge&logo=ros&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Turtlebot4](https://img.shields.io/badge/Platform-Turtlebot4-green?style=for-the-badge)
![YOLO](https://img.shields.io/badge/AI-YOLOv8-FF0000?style=for-the-badge)
![Nav2](https://img.shields.io/badge/Navigation-Nav2-orange?style=for-the-badge)

> **"직원 대신 로봇이 옷을 찾아드립니다."**
> 
> 의류 매장을 방문한 손님이 웹으로 옷을 주문하면, **2대의 Turtlebot4 로봇**이 스스로 창고로 이동하여 재고를 파악하고 손님에게 직접 배달하는 **O2O(Online to Offline) 매장 자동화 시스템**입니다.

<br>

[🎬 데모 영상 보러가기](https://drive.google.com/file/d/1rgOwfXtNjZ27KaW0gpJynZV6yzTcaGBK/view?usp=sharing)

## 🏪 프로젝트 시나리오 (Service Scenario)

이 시스템은 복잡한 의류 매장 환경에서 직원의 반복적인 물류 업무를 자동화하기 위해 설계되었습니다.

1.  **Order (주문):** 손님이 매장 내 키오스크/모바일 웹에서 원하는 의류를 주문합니다.
2.  **Dispatch (배차):** 대기 중인 로봇이 주문 정보를 수신하고, 매장 지도를 기반으로(SLAM) 창고로 이동합니다.
3.  **Inventory Check (재고 파악):** 로봇이 지정된 선반 앞에서 **YOLOv8** 비전 시스템을 이용해 해당 의류(Box)의 재고 유무를 확인합니다.
4.  **Delivery (배송):** 물품을 싣고 매장 내 진열대와 사람을 피해 손님 위치로 정확하게 배달합니다.
5.  **Return (복귀):** 임무를 마친 로봇은 자동으로 도킹 스테이션으로 복귀하여 충전합니다.

<br>

## 🏛️ 시스템 아키텍처 (System Architecture)

```mermaid
graph LR
    A["👤 손님 (Web Client)"] -->|의류 주문| B("☁️ 통합 관제 서버 (FastAPI)")
    B -->|"재고 확인 요청"| C["🐢 Mission Master Node"]
    
    subgraph Robot System [Turtlebot4 Swarm]
        C -->|"Nav2 Action"| D["🤖 Robot 2 (Leader)"]
        C -->|"Nav2 Action"| E["🤖 Robot 3 (Follower)"]
        D & E -->|"Traffic Control"| F{"🚦 충돌 방지 로직"}
        D & E -->|"Object Detection"| G["📷 OAK-D Lite (YOLO)"]
    end

    G -->|"재고 데이터 전송"| B
    F -->|"안전 주행"| H["🏁 고객 위치 도착"]
```
---

## 🤖 핵심 기능 1: 멀티 로봇 자율 협업 (Robot Intelligence)

본 프로젝트의 핵심 파트로, 좁은 창고 환경에서 다수의 로봇이 충돌 없이 임무를 완수하기 위한 고도화된 로직이 적용되었습니다.

1. 단계별 미션 수행 (Phase System)
로봇의 행동을 5단계로 세분화하여 체계적으로 관리합니다.

| 단계 | Phase Name | 설명 |
|----|-----------|------|
| 1 | Entry | 대기 장소에서 창고 진입로(`goal_1`)로 이동 (Nav2) |
| 2 | Approach | 박스 적재 위치 바로 앞으로 정밀 접근 (Nudge Control) |
| 3 | Detection | 정지 후 YOLOv8로 박스 개수 카운트 및 ROI 점유 신호 발신 |
| 4 | Delivery | 배달 위치로 이동 후 서버(DB)에 재고 데이터 전송 및 알림음 재생 |
| 5 | Return | 작업 완료 후 도킹 스테이션 복귀 및 자동 충전 (Auto Docking) |

2. 스마트 교통 정리 (Smart Traffic Control)
로봇 간 충돌 방지를 위해 이중 확인(Dual Check) 로직을 구현했습니다.

Phase Sharing: /current_phase 토픽을 통해 서로의 작업 단계를 실시간 공유합니다.

ROI Occupancy: 작업 구역(ROI) 진입 시 /ROI_robot_detected (Bool) 신호를 송출합니다.

Smart Wait: 후행 로봇은 선행 로봇이 위험 구간(Phase 1~3)에 있을 경우, 지정된 대기 장소로 회피하여 10초간 대기합니다.

3. Lidar 기반 안전 가드 (Safety Monitor)
Nav2의 Costmap과는 별개로, Raw Lidar 데이터를 직접 구독하여 비상 상황에 대처합니다.

Emergency Stop: 전방 30도, 0.4m 이내 장애물 감지 시 즉시 정지.

Active Avoidance: 정지 후 전방 100도 범위를 스캔하여 더 넓은 공간으로 회피 기동 수행.

## 🖥️ 핵심 기능 2: 통합 관제 웹 플랫폼 (Web Control)
로봇이 수집한 정보를 시각화하고 제어하는 사용자 인터페이스입니다.

실시간 대시보드: 다중 로봇의 위치(Odom), 배터리 상태, 현재 작업 Phase를 0.5초 단위로 모니터링.

원스톱 주문 시스템: 사용자가 물품을 주문하면 가용 로봇을 자동으로 배차하고 자율주행 명령 전송.

재고 데이터 동기화: 로봇이 Phase 3에서 파악한 박스 개수가 Phase 4 완료 시점에 DB에 자동 업데이트됨.

---

## 🔥 기술적 도전 및 해결 (Troubleshooting)
### 🚀 Robot Side Issues
1. Ghost Arrival 현상: goToPose 명령 직후 이동하지 않았는데 '도착 성공' 처리되는 문제.

👉 해결: 명령 전송 후 time.sleep(1.5) 강제 부여 및 distance_remaining 이중 체크 로직 추가.

2. Initial Pose 오차: 하드코딩된 초기 좌표와 실제 맵의 미세한 불일치로 인한 주행 오류.

👉 해결: setInitialPose 강제 설정을 제거하고, **"도킹 스테이션 = (0,0) = North"**라는 물리적 기준을 활용하여 Undock 후 주행 시작.

3. 벽 앞 진동(Oscillation): 좁은 회피 각도로 인해 장애물 앞에서 로봇이 떨기만 하는 현상.

👉 해결: 회피 판단 시야각을 30도에서 100도로 확장하여 탈출 경로 확보.

### ⚡ Web Side Issues
1. FastAPI & ROS2 충돌: 단일 스레드에서 웹 서버와 ROS 통신 동시 실행 시 블로킹 발생.

👉 해결: Startup Event에서 ROS2 노드를 **데몬 스레드(Daemon Thread)**로 분리하여 비동기 처리.

---

## 🛠️ 기술 스택 (Tech Stack)
| 구분 | 기술 스택 |
|----|-----------|
| Robotics | ROS 2 Humble, Nav2, Gazebo, Turtlebot4 |
| AI / Vision | YOLOv8 (Custom Trained), OpenCV, OAK-D Lite |
| Backend | Python 3.10, FastAPI, MySQL, SQLAlchemy |
| Frontend | React, JavaScript, HTML5 / CSS3 |
| Infra | Ubuntu 22.04 LTS |

---

## 📦 설치 및 실행 (How to Run)
1. 로봇 제어 노드 (Robot Side)
```bash
# 의존성 패키지 설치
sudo apt install ros-humble-turtlebot4-navigation ros-humble-nav2-simple-commander
pip install ultralytics opencv-python

# 패키지 빌드 및 실행
cd ~/rokey_ws
colcon build --symlink-install --packages-selec Picker_project
source install/setup.bash
ros2 run Picker_project picker_main
```
2. 웹 서버 및 클라이언트 (Web Side)
```bash
# Backend 실행
cd backend
pip install -r requirements.txt
python main.py

# Frontend 실행
cd frontend
npm install
npm start
```

---

## License
