# syntax=docker/dockerfile:1.7
# rover_sim — standalone Docker build of the mighty ground-robot sim stack:
# mighty + mpc + global_mapper_ros (acl-mapping) + Gazebo/RViz/Livox-sim, all
# pulled from their upstream repos (rover_sim.repos + the *_GIT_* args below)
# rather than requiring a local checkout of any of them.
#
# Stages:
#   *-src         mighty, acl-mapping, mpc and elevation_mapping_cupy at their
#                 pinned commits. Each can be replaced by a local checkout:
#                 --build-context <name>-src=<dir>
#                 (make build MIGHTY_SRC=... ACL_MAPPING_SRC=... MPC_SRC=... ELEVATION_SRC=...)
#   deps-src      clone the rest of rover_sim.repos
#   deps-build    rosdep + colcon the deps, plus the two non-colcon Livox builds
#   overlay-build acl-mapping + mpc on top of the deps
#   mighty-build  mighty on top of the deps, colcon BUILD_SIMULATION=ON
#   rover-sim-build this repo's rover_sim package (robot models, spawn launch)
#   elevation-build elevation_mapping_cupy (the rovers' mapper)
#   runtime       exec deps + Gazebo/RViz + the install trees (ACL mapper)
#   runtime-elevation runtime + elevation_mapping_cupy and its CUDA Python deps
#                 (make build MAPPER=elevation; needs an NVIDIA GPU to run)
#
# Needs SSH for the private repos (make build forwards the agent's key).
ARG WS=/home/swarm/code/rover_sim_ws
# Caps colcon/make/ninja parallelism. Uncapped, package-level parallelism
# stacks with each package's own thread count and can exhaust RAM on a build
# host with heavy template-instantiation packages (PCL/Eigen/Gazebo).
# --build-arg JOBS=N to change.
ARG JOBS=1
ARG MIGHTY_GIT_URL=git@github.com:blakete/mighty.git
ARG MIGHTY_GIT_REF=6caed3835fad7be951fb1715b59bec7dab2de49c
ARG ACL_MAPPING_GIT_URL=git@gitlab.com:seraHam/acl-mapping.git
ARG ACL_MAPPING_GIT_REF=349a77b226e5b42fc93df1c15086a124ea711564
ARG MPC_GIT_URL=git@gitlab.com:mit-acl/ugv/ugv_control/mpc.git
ARG MPC_GIT_REF=7efbfe5ff76d70a332287f5b884248c29f197cba
# The commit the rovers run.
ARG ELEVATION_GIT_URL=https://github.com/nehirgerek/elevation_mapping_cupy.git
ARG ELEVATION_GIT_REF=3968f412e799098b63d7dc3d0def30b784c378f5
ARG ROS2_NUMPY_GIT_URL=https://github.com/Box-Robotics/ros2_numpy.git
ARG ROS2_NUMPY_GIT_REF=d5ebf723e1d2457d36fa89bf021b212e340c3c11

# ---------- *-src: the repos a local checkout can replace -------------------
FROM ros:humble-ros-base AS git-base
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client git \
    && rm -rf /var/lib/apt/lists/*

FROM git-base AS mighty-git
ARG MIGHTY_GIT_URL
ARG MIGHTY_GIT_REF
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -H github.com gitlab.com >> ~/.ssh/known_hosts 2>/dev/null \
 && git clone "${MIGHTY_GIT_URL}" /src && git -C /src checkout "${MIGHTY_GIT_REF}"
FROM scratch AS mighty-src
COPY --from=mighty-git /src /

FROM git-base AS acl-mapping-git
ARG ACL_MAPPING_GIT_URL
ARG ACL_MAPPING_GIT_REF
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -H github.com gitlab.com >> ~/.ssh/known_hosts 2>/dev/null \
 && git clone "${ACL_MAPPING_GIT_URL}" /src && git -C /src checkout "${ACL_MAPPING_GIT_REF}"
FROM scratch AS acl-mapping-src
COPY --from=acl-mapping-git /src /

FROM git-base AS mpc-git
ARG MPC_GIT_URL
ARG MPC_GIT_REF
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -H github.com gitlab.com >> ~/.ssh/known_hosts 2>/dev/null \
 && git clone "${MPC_GIT_URL}" /src && git -C /src checkout "${MPC_GIT_REF}"
FROM scratch AS mpc-src
COPY --from=mpc-git /src /

FROM git-base AS elevation-git
ARG ELEVATION_GIT_URL
ARG ELEVATION_GIT_REF
RUN git clone "${ELEVATION_GIT_URL}" /src && git -C /src checkout "${ELEVATION_GIT_REF}"
FROM scratch AS elevation-src
COPY --from=elevation-git /src /

# ---------- deps-src: every other pinned dependency source -------------------
FROM ros:humble-ros-base AS deps-src
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client python3-vcstool \
    && rm -rf /var/lib/apt/lists/*
COPY rover_sim.repos /tmp/rover_sim.repos
# --mount=type=ssh forwards the caller's key for the private repos.
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -H github.com gitlab.com >> ~/.ssh/known_hosts 2>/dev/null \
 && mkdir -p ${WS}/src \
 && vcs import ${WS}/src < /tmp/rover_sim.repos

# ---------- deps-build: rosdep + colcon the dependency workspace ------------
FROM ros:humble-ros-base AS deps-build
ARG WS
ARG JOBS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# MAKEFLAGS covers GNU Make; CMAKE_BUILD_PARALLEL_LEVEL is generator-agnostic
# (some packages here pull in Ninja, which ignores MAKEFLAGS). Inherited by
# overlay-build and mighty-build via FROM deps-build.
ENV MAKEFLAGS=-j${JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${JOBS}
WORKDIR ${WS}
COPY --from=deps-src ${WS}/src ${WS}/src

# Livox-SDK2/livox_ros_driver2 aren't plain colcon packages (see below) —
# pull them out before rosdep/colcon see them.
RUN mkdir -p /opt/livox_sdk /opt/livox_ws/src \
 && mv ${WS}/src/Livox-SDK2 /opt/livox_sdk/src \
 && mv ${WS}/src/livox_ros_driver2 /opt/livox_ws/src/livox_ros_driver2

# --skip-keys: ament_python is mpc's buildtool_depend, not a rosdistro key.
# catkin: DecompUtil's package.xml says buildtool_depend catkin but it's a
# plain CMake project. Eigen3/Armadillo/Boost/OpenCV/PCL/livox_ros_driver2/
# tf/dynamic_reconfigure: aerial-only leftovers in uav_simulator declare these
# as rosdep keys, wrong-cased or ROS1-only — none resolve via rosdep; install
# the real libs directly and skip the bogus keys instead.
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src \
        --skip-keys "catkin ament_python Eigen3 Armadillo Boost OpenCV PCL livox_ros_driver2 tf dynamic_reconfigure" \
 && apt-get install -y --no-install-recommends \
        libeigen3-dev libarmadillo-dev libboost-all-dev libpcl-dev libopencv-dev \
        ros-humble-rviz2 ros-humble-rviz-common \
        libgl1-mesa-glx libgl1-mesa-dri mesa-utils \
 && rm -rf /var/lib/apt/lists/*

# Livox-SDK2: plain CMake project, cmake-installed to /usr/local.
RUN cd /opt/livox_sdk/src && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"${JOBS}" && make install \
 && ldconfig

# livox_ros_driver2: built via its own build.sh (picks a message-format
# variant), in its own workspace so it doesn't collide with ${WS}'s colcon build.
RUN source /opt/ros/humble/setup.bash \
 && cd /opt/livox_ws/src/livox_ros_driver2 && ./build.sh humble

ENV CMAKE_PREFIX_PATH=/opt/livox_ws/install/livox_ros_driver2:${CMAKE_PREFIX_PATH} \
    LD_LIBRARY_PATH=/opt/livox_ws/install/livox_ros_driver2/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# decomp_util first: decomp_rviz_plugins find_package()s it without declaring
# the dependency, so colcon can't order them in one invocation.
RUN source /opt/ros/humble/setup.bash \
 && colcon build --executor sequential --packages-select decomp_util --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && source install/setup.bash \
 && colcon build --executor sequential --packages-skip decomp_util --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && rm -rf build log

# ---------- overlay-build: acl-mapping + mpc on top of the deps ---------------
FROM deps-build AS overlay-build
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WS}
COPY --from=acl-mapping-src / ${WS}/src/acl-mapping
COPY --from=mpc-src / ${WS}/src/mpc
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src \
        --skip-keys "catkin ament_python Eigen3 Armadillo Boost OpenCV PCL livox_ros_driver2 tf dynamic_reconfigure" \
 && rm -rf /var/lib/apt/lists/*
# Only these packages' install dirs go to the runtime stage (the deps come from mighty-build).
RUN source /opt/ros/humble/setup.bash && source install/setup.bash \
 && pkgs=$(colcon list --names-only --base-paths src/acl-mapping src/mpc) \
 && colcon build --executor sequential --base-paths src/acl-mapping src/mpc \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && mkdir -p /overlay-install && for p in $pkgs; do cp -a install/$p /overlay-install/; done \
 && rm -rf build log
RUN mkdir -p /manifests/overlay && cd src \
 && find acl-mapping mpc -name package.xml -exec cp --parents {} /manifests/overlay \; \
 && find /manifests -exec touch -d @0 {} +

# ---------- mighty-build: mighty on top of the deps --------------------------
FROM deps-build AS mighty-build
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WS}
COPY --from=mighty-src / ${WS}/src/mighty
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src \
        --skip-keys "catkin ament_python Eigen3 Armadillo Boost OpenCV PCL livox_ros_driver2 tf dynamic_reconfigure mpc" \
 && apt-get install -y --no-install-recommends nlohmann-json3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN source /opt/ros/humble/setup.bash && source install/setup.bash \
 && colcon build --executor sequential --packages-select mighty \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_SIMULATION=ON -DBUILD_TESTING=OFF \
 && rm -rf build log
# Manifests for the runtime stage's rosdep, mtime-pinned so this stays cached
# across source changes that leave the package.xml files alone.
RUN mkdir -p /manifests && cd src \
 && find . -name package.xml -exec cp --parents {} /manifests \; \
 && find /manifests -exec touch -d @0 {} +

# ---------- rover-sim-build: this repo's rover_sim package ---------------------
FROM ros:humble-ros-base AS rover-sim-build
ARG WS
SHELL ["/bin/bash", "-c"]
WORKDIR ${WS}
COPY src/rover_sim ${WS}/src/rover_sim
RUN source /opt/ros/humble/setup.bash \
 && colcon build --packages-select rover_sim \
 && rm -rf build log

# ---------- elevation-build: elevation_mapping_cupy + ros2_numpy -------------
FROM git-base AS elevation-build
ARG WS
ARG ROS2_NUMPY_GIT_URL
ARG ROS2_NUMPY_GIT_REF
WORKDIR ${WS}
COPY --from=elevation-src / ${WS}/src/elevation_mapping_cupy
# sensor_processing (semantic_sensor, for camera semantics) isn't used.
RUN git clone "${ROS2_NUMPY_GIT_URL}" src/ros2_numpy && git -C src/ros2_numpy checkout "${ROS2_NUMPY_GIT_REF}" \
 && touch src/elevation_mapping_cupy/sensor_processing/COLCON_IGNORE \
 && apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src --skip-keys "semantic_sensor" \
 && rm -rf /var/lib/apt/lists/*
RUN source /opt/ros/humble/setup.bash \
 && pkgs=$(colcon list --names-only --base-paths src) \
 && colcon build --base-paths src --cmake-args -DBUILD_TESTING=OFF \
 && mkdir -p /elevation-install && for p in $pkgs; do cp -a install/$p /elevation-install/; done \
 && rm -rf build log
RUN mkdir -p /manifests && cd src \
 && find . -name package.xml -not -path "*/sensor_processing/*" -not -path "*/plane_segmentation/*" \
        -exec cp --parents {} /manifests \; \
 && find /manifests -exec touch -d @0 {} +

# ---------- runtime: exec deps + Gazebo/RViz + the baked workspace -----------
FROM ros:humble-ros-base AS runtime
LABEL rover_sim.mappers="acl"
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WS}
COPY --from=mighty-build /manifests /tmp/manifests
COPY --from=overlay-build /manifests/overlay /tmp/manifests/overlay

# Stock Mesa: uses the GPU when it supports it, otherwise llvmpipe. GPUs newer
# than jammy's Mesa need a newer one (e.g. the kisak/turtle PPA), added locally.

RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --dependency-types exec --from-paths /tmp/manifests \
        --skip-keys "catkin ament_python Eigen3 Armadillo Boost OpenCV PCL livox_ros_driver2 tf dynamic_reconfigure" \
 && apt-get install -y --no-install-recommends \
        ros-humble-rviz2 ros-humble-rviz-common ros-humble-rqt-gui ros-humble-rqt-gui-py \
        libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers mesa-utils \
        libeigen3-dev libarmadillo-dev libboost-all-dev libpcl-dev libopencv-dev \
        tmux nano python3-pip ros-humble-xacro \
 && rm -rf /var/lib/apt/lists/* /tmp/manifests

# mpc_node imports casadi/do_mpc.
RUN pip3 install --no-cache-dir casadi==3.6.7 do-mpc==5.1.1 \
 && pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.3

# tmuxp: for the multi-pane sessions in launch/.
RUN pip3 install --no-cache-dir tmuxp

# Livox SDK runtime libs + the separately-built driver package.
COPY --from=deps-build /usr/local /usr/local
COPY --from=deps-build /opt/livox_ws/install /opt/livox_ws/install
RUN ldconfig
ENV LD_LIBRARY_PATH=/opt/livox_ws/install/livox_ros_driver2/lib:/usr/local/lib:${LD_LIBRARY_PATH}

COPY --from=mighty-build ${WS}/install ${WS}/install
COPY --from=overlay-build /overlay-install ${WS}/install
COPY --from=rover-sim-build ${WS}/install/rover_sim ${WS}/install/rover_sim
# Lets Gazebo resolve package://rover_sim/... and package://mighty/... meshes.
ENV GAZEBO_MODEL_PATH=${WS}/install/rover_sim/share:${WS}/install/mighty/share
# Baked fallback .bashrc for running standalone without compose. Under the
# normal make workflow, compose.yaml mounts docker/dotfiles/bashrc over
# this same path instead, which is what actually applies.
RUN printf 'source /opt/ros/humble/setup.bash\nsource %s/install/setup.bash\nsource /opt/livox_ws/install/setup.bash\n' "${WS}" >> /root/.bashrc
# Command comes from compose: container just idles (sleep infinity); nodes
# are docker-exec'd in or run from an attached shell.

# For stacks run in this container: nav2's map_server (a known map) and the RViz
# 2D overlay messages (on-screen text).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager \
        ros-humble-rviz-2d-overlay-msgs \
 && rm -rf /var/lib/apt/lists/*

# ---------- runtime-elevation: + elevation_mapping_cupy (MAPPER=elevation) ----
FROM runtime AS runtime-elevation
ARG WS
LABEL rover_sim.mappers="acl elevation"
COPY --from=elevation-build /manifests /tmp/manifests
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --dependency-types exec --from-paths /tmp/manifests --skip-keys "semantic_sensor" \
 && rm -rf /var/lib/apt/lists/* /tmp/manifests
# The rovers' Python deps. CUDA comes from pip (12.4, what l01's 550 driver
# supports): torch's cu124 wheels and CuPy's cuda-toolkit extra. numpy 2 (CuPy
# 14 needs it) breaks jammy's apt cv2, so OpenCV comes from pip too.
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0+cu124 \
        cupy-cuda12x==14.1.1 "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==12.4.1" \
        opencv-python-headless==5.0.0.93 simple-parsing==0.1.9 shapely==2.1.2 \
        transforms3d==0.4.2 ruamel.yaml==0.19.1 numpy==2.2.6 scipy==1.15.3
# CuPy's cupyx.cusolver links libcusolver.so.11 by name, so the pip CUDA
# libraries have to be on the loader path.
ARG NV=/usr/local/lib/python3.10/dist-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cusolver/lib:${NV}/cublas/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${LD_LIBRARY_PATH}
COPY --from=elevation-build /elevation-install ${WS}/install
