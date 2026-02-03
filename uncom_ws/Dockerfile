FROM palroboticssl/tiago_tutorials:noetic
ENTRYPOINT [ ]
ENV DEBIAN_FRONTEND=noninteractive

# 1. Fix GPG Keys and Install Dependencies (Combined to save layers)
RUN apt-get update && apt-get install -y curl && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ros-noetic-moveit \
    ros-noetic-vision-msgs \
    ros-noetic-tf2-sensor-msgs \
    python3-catkin-tools \
    mosquitto\
    python3-pip\
    libgl1-mesa-glx \
    libgomp1 \
    libusb-1.0-0 \
    git && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# 2. Setup Workspace
RUN mkdir -p /root/output  # directory of media files for uncom
RUN mkdir -p /root/catkin_ws/src
WORKDIR /root/catkin_ws/src

RUN pip install webrtcvad paho-mqtt open3d

# 3. Clone Repos
RUN git clone https://github.com/ichores-research/motion_msgs.git
RUN git clone https://github.com/v4r-tuwien/object_detector_msgs.git

# 4. BUILD - THIS IS THE IMPORTANT PART
# You MUST source the TIAGo workspace HERE so catkin sees 'tiago_description'
RUN source /opt/ros/noetic/setup.bash && \
    source /tiago_public_ws/devel/setup.bash && \
    cd /root/catkin_ws && \
    catkin build

# 5. Add sourcing to bashrc for interactive debugging
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /tiago_public_ws/devel/setup.bash" >> ~/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc

WORKDIR /root

# 6. Run Command
CMD [ "bash", "-c", "source /opt/ros/noetic/setup.bash && source /tiago_public_ws/devel/setup.bash && source /root/catkin_ws/devel/setup.bash]
