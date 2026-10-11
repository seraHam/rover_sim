# syntax=docker/dockerfile:1.7
# rover_sim — standalone Docker build of the mighty ground-robot sim stack:
# mighty + mpc + global_mapper_ros (acl-mapping) + Gazebo/RViz/Livox-sim, all
# pulled from their upstream repos (rover_sim.repos + the MIGHTY_GIT_* args
# below) rather than requiring a local checkout of any of them.
#
# Stages:
#   deps-src     clone every rover_sim.repos entry (dependencies only)
#   deps-build   rosdep + colcon the deps, plus the two non-colcon Livox builds
#   mighty-build clone mighty itself (separate from deps-src, see
#                rover_sim.repos' comment on why), colcon BUILD_SIMULATION=ON
#   runtime      exec deps + Gazebo/RViz + the install tree
#
# Needs SSH for the private mpc/acl-mapping repos and mighty itself:
#   docker compose -f docker/compose.yaml build --ssh default[=~/.ssh/<key>]
# (make build does this; see Makefile).
ARG WS=/home/swarm/code/mighty_ws
# Caps colcon/make/ninja parallelism. Uncapped, package-level parallelism
# stacks with each package's own thread count and can exhaust RAM on a build
# host with heavy template-instantiation packages (PCL/Eigen/Gazebo).
# --build-arg JOBS=N to change.
ARG JOBS=1
# mighty itself — deliberately not in rover_sim.repos, see its header comment.
ARG MIGHTY_GIT_URL=git@github.com:blakete/mighty.git
ARG MIGHTY_GIT_REF=68a09873b3cd54a47f2fb46a85336d4081c9b634

# ---------- deps-src: every pinned dependency source (not mighty itself) ----
FROM ros:humble-ros-base AS deps-src
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-client python3-vcstool \
    && rm -rf /var/lib/apt/lists/*
COPY rover_sim.repos /tmp/rover_sim.repos
# --mount=type=ssh forwards the caller's key for the private mpc/acl-mapping repos.
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
# mighty-build via FROM deps-build.
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

# ---------- mighty-build: mighty itself, cloned separately, on top of deps --
FROM deps-build AS mighty-build
ARG WS
ARG MIGHTY_GIT_URL
ARG MIGHTY_GIT_REF
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WS}
RUN --mount=type=ssh \
    mkdir -p -m 0700 ~/.ssh && ssh-keyscan -H github.com gitlab.com >> ~/.ssh/known_hosts 2>/dev/null \
 && git clone "${MIGHTY_GIT_URL}" ${WS}/src/mighty \
 && git -C ${WS}/src/mighty checkout "${MIGHTY_GIT_REF}"
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src \
        --skip-keys "catkin ament_python Eigen3 Armadillo Boost OpenCV PCL livox_ros_driver2 tf dynamic_reconfigure" \
 && apt-get install -y --no-install-recommends nlohmann-json3-dev \
 && rm -rf /var/lib/apt/lists/*
RUN source /opt/ros/humble/setup.bash && source install/setup.bash \
 && colcon build --executor sequential --packages-select mighty \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_SIMULATION=ON -DBUILD_TESTING=OFF \
 && rm -rf build log
# Manifests for the runtime stage's rosdep, mtime-pinned so this stays cached
# across mighty pin bumps.
RUN mkdir -p /manifests && cd src \
 && find . -name package.xml -exec cp --parents {} /manifests \; \
 && find /manifests -exec touch -d @0 {} +

# ---------- runtime: exec deps + Gazebo/RViz + the baked workspace -----------
FROM ros:humble-ros-base
ARG WS
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${WS}
COPY --from=mighty-build /manifests /tmp/manifests

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
# Baked fallback .bashrc for running standalone without compose. Under the
# normal make workflow, compose.yaml mounts docker/dotfiles/bashrc over
# this same path instead, which is what actually applies.
RUN printf 'source /opt/ros/humble/setup.bash\nsource %s/install/setup.bash\nsource /opt/livox_ws/install/setup.bash\n' "${WS}" >> /root/.bashrc
# Command comes from compose: container just idles (sleep infinity); nodes
# are docker-exec'd in or run from an attached shell.
