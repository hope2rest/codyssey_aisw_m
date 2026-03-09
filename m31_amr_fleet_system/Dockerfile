# =============================================================================
# AMR Fleet System Docker 이미지
# =============================================================================
FROM ros:humble

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-robot-localization \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
RUN pip3 install --no-cache-dir \
    numpy>=1.24 \
    scipy>=1.10 \
    ultralytics>=8.0 \
    pyyaml>=6.0 \
    matplotlib>=3.7

# 워크스페이스 생성
ENV WORKSPACE=/ros2_ws
RUN mkdir -p ${WORKSPACE}/src
WORKDIR ${WORKSPACE}

# 소스 복사
COPY . ${WORKSPACE}/src/amr_fleet_system/

# ROS2 워크스페이스 빌드
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# 엔트리포인트
COPY <<'ENTRYPOINT_SCRIPT' /entrypoint.sh
#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
exec "$@"
ENTRYPOINT_SCRIPT
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
