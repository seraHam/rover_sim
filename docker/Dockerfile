# syntax=docker/dockerfile:1.7
# rover_sim — Docker builds for the simulation. Two independent sets of targets
# share this file until the legacy one is removed (Phase 6).
#
# STACK TARGETS (docker/compose.stack.yaml, built by scripts/build_images.sh).
# The JR02 stacks (mighty, mapping_ws, dlio_ws, rover) are built from each
# repo's own Dockerfile at the pins in docker/stacks.env; only the containers
# with no robot image are built here:
#   stack-base  ros:humble-ros-base pinned by digest + the fleet's rmw_zenoh pin
#   gazebo      Gazebo classic + gazebo_ros, the Livox sim plugin and its msg
#               types, decomp's RViz plugins, RViz, and this repo's rover_sim
#               package. Sim-only deps: rover_sim.sim.repos, or a local vcs
#               import of it with --build-context stack-sim-src=<dir>
#   elevation   elevation_mapping_cupy + ros2_numpy and their CUDA Python deps
#               (the robot runs this natively on its Orin; needs an NVIDIA GPU
#               to run). Local checkout: --build-context stack-elevation-src=<dir>
# Their helper stages are named stack-* and share nothing with the legacy
# stages. Every source they clone is public, so they need no SSH.
#
# LEGACY MONOLITH (rover-sim:local; docker/compose.yaml, make build/up):
# standalone build of the mighty ground-robot sim stack:
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

# ---------- stack targets: pins ----------------------------------------------
# Base of every stack-* stage, pinned by digest so all of them, and every
# rebuild, start from the same ROS packages (the legacy stages float on the
# tag). This is the 2026-09-09 build (rclcpp 16.0.19). JR02's images came from
# another ros:humble-ros-base build (sha256:860e5379..., seen only in its
# BuildKit cache; it is neither this index nor one of its platform images);
# move this pin to it once its full digest is known.
ARG STACK_BASE_IMAGE=ros:humble-ros-base@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138
# elevation target: the fleet's fork (parameter overlays, extra cloud inputs).
# The legacy ELEVATION_GIT_* above are the older nehirgerek commit.
ARG STACK_ELEVATION_GIT_URL=https://github.com/blakete/elevation_mapping_cupy.git
ARG STACK_ELEVATION_GIT_REF=43803679774cd04b36007aca84577b74999e9771
ARG STACK_ROS2_NUMPY_GIT_URL=https://github.com/Box-Robotics/ros2_numpy.git
ARG STACK_ROS2_NUMPY_GIT_REF=d5ebf723e1d2457d36fa89bf021b212e340c3c11

# ---------- stack-base: pinned base + the fleet's rmw_zenoh build ------------
FROM ${STACK_BASE_IMAGE} AS stack-base
ARG STACK_BASE_IMAGE
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# The block below is mighty 3cf7320 docker/Dockerfile.hw:135-193 (the robot
# images' pin), minus its rosdep line. In the sim it puts the rover_sim-built
# containers on the same rmw_zenoh build as the robot-built ones.
#
# RMW PIN. Every container on a rover (drive, sensors, dlio, this one) must run the
# SAME rmw_zenoh build. Two builds that both call themselves 0.1.9 are not
# interchangeable: with dlio/drive/sensors on 0.1.9-1jammy.20260723 and this image
# on 0.1.9-1jammy.20260907 (what an unpinned install pulled on 2026-09-07),
# TRANSIENT_LOCAL subscriptions across that boundary received NOTHING while
# VOLATILE ones were fine. tf2 takes /tf_static as TRANSIENT_LOCAL, so MPC never
# saw <ns>/map -> <ns>/odom, commanded 0/0 in AUTO, and the rover sat still with a
# plan on screen (RR08, 2026-09-17). packages.ros.org keeps only the newest build
# per version, so the pin comes from the dated ROS snapshot repo instead, which is
# signed by the same ROS archive key the base image already ships in
# /usr/share/keyrings/ros2-archive-keyring.gpg.
# apt-mark hold + the dpkg-query at the end make a silent drift fail the build.
# When the fleet moves forward, bump all three ARGs together and rebuild EVERY
# rover image in the same rollout.
#
# 2026-09-19: the fleet pin moved to rmw_zenoh 0.1.9-1jammy.20260907.200640 (the
# build the C2 cart and every rover drive/sensors/dlio image now carry -- rover repo
# rover_bringup/docker/Dockerfile{,.sensors}, dlio_ws docker/Dockerfile). No
# snapshots.ros.org snapshot carries that build yet (newest is 2026-08-07 =
# 20260723), so RMW_ZENOH_CPP_VER resolves from stock packages.ros.org, which the
# base image already lists; the snapshot line below still serves the vendor pin
# (20260723.020902 is identical on both). packages.ros.org keeps only the newest
# build per version, so this exact pin stops resolving once ROS rebuilds 0.1.9 --
# the dpkg-query tripwire below fails the build loudly when that happens; move
# every fleet image together (or re-pin to a snapshot once one carries 20260907).
#
# 2026-10-12: that happened. packages.ros.org moved on to 0.1.11 (20261011) and dropped the
# 20260907 build, so the build failed at the apt install. The 2026-09-14 snapshot carries both
# pins exactly (rmw_zenoh 0.1.9-1jammy.20260907.200640, zenoh_cpp_vendor
# 0.1.9-1jammy.20260723.020902), so ROS_SNAPSHOT below now serves both and nothing depends on
# packages.ros.org. Snapshots are immutable, so this no longer rots.
ARG ROS_SNAPSHOT=2026-09-14
ARG RMW_ZENOH_CPP_VER=0.1.9-1jammy.20260907.200640
ARG ZENOH_CPP_VENDOR_VER=0.1.9-1jammy.20260723.020902
# snapshots.ros.org is signed by "ROS Snapshot builder" (AD19BAB3CBF125EA), NOT by the
# Open Robotics key in ros2-archive-keyring.gpg that packages.ros.org uses. Vendored
# here rather than pulled from a keyserver so the build needs no keyserver and cannot
# silently pick up a different key; the fingerprint assert below is the tripwire.
ARG ROS_SNAPSHOT_KEY_FPR=4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA
COPY docker/keys/ros-snapshot-archive-keyring.asc /tmp/ros-snapshot.asc
RUN gpg --dearmor < /tmp/ros-snapshot.asc > /usr/share/keyrings/ros-snapshot-archive-keyring.gpg \
 && gpg --show-keys --with-colons /tmp/ros-snapshot.asc | awk -F: '/^fpr/{print $10; exit}' \
      | grep -qx "${ROS_SNAPSHOT_KEY_FPR}" \
 && rm -f /tmp/ros-snapshot.asc \
 && echo "deb [signed-by=/usr/share/keyrings/ros-snapshot-archive-keyring.gpg] http://snapshots.ros.org/humble/${ROS_SNAPSHOT}/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2-snapshot.list \
 && printf 'Package: ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor\nPin: origin snapshots.ros.org\nPin-Priority: 1001\n' \
        > /etc/apt/preferences.d/99-rmw-zenoh-pin \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        ros-humble-rmw-zenoh-cpp=${RMW_ZENOH_CPP_VER} \
        ros-humble-zenoh-cpp-vendor=${ZENOH_CPP_VENDOR_VER} \
        tmux python3-pip \
 && apt-mark hold ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor \
 && test "$(dpkg-query -W -f='${Version}' ros-humble-rmw-zenoh-cpp)" = "${RMW_ZENOH_CPP_VER}" \
 && test "$(dpkg-query -W -f='${Version}' ros-humble-zenoh-cpp-vendor)" = "${ZENOH_CPP_VENDOR_VER}" \
 && rm -rf /var/lib/apt/lists/*
# The verified pin, for the targets below to re-check after their own apt
# layers (hold + pin make a drift unlikely; this fails the layer that caused it):
#   dpkg-query -W ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor | diff /etc/rover_sim/rmw-zenoh.pin -
RUN mkdir -p /etc/rover_sim \
 && dpkg-query -W ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor > /etc/rover_sim/rmw-zenoh.pin
LABEL org.opencontainers.image.base.name="${STACK_BASE_IMAGE}" \
      rover_sim.rmw_zenoh_cpp="${RMW_ZENOH_CPP_VER}" \
      rover_sim.zenoh_cpp_vendor="${ZENOH_CPP_VENDOR_VER}"

# ---------- stack-sim-src: the gazebo target's sim-only deps -----------------
FROM stack-base AS stack-sim-git
COPY rover_sim.sim.repos /tmp/rover_sim.sim.repos
RUN mkdir -p /src && vcs import /src < /tmp/rover_sim.sim.repos
FROM scratch AS stack-sim-src
COPY --from=stack-sim-git /src /

# ---------- stack-sim-manifests: their package.xml files, for rosdep ---------
# Both rosdep installs (build deps below, exec deps in gazebo) read only these,
# mtime-pinned (as in mighty-build), so a pin bump that leaves a repo's
# dependencies alone rebuilds its packages without re-running the apt layers.
# livox_ros_driver2's ROS 2 manifest is package_ROS2.xml (its build.sh copies
# it into place); Livox-SDK2 is plain CMake.
FROM stack-base AS stack-sim-manifests
COPY --from=stack-sim-src / /src
RUN mkdir -p /manifests/livox_ros_driver2 && cd /src \
 && find . \( -path ./livox_ros_driver2 -o -path ./Livox-SDK2 \) -prune -o -name package.xml \
        -exec cp --parents {} /manifests \; \
 && cp livox_ros_driver2/package_ROS2.xml /manifests/livox_ros_driver2/package.xml \
 && find /manifests -exec touch -d @0 {} +

# ---------- stack-gazebo-build: gazebo_ros, Livox sim + msgs, decomp ---------
FROM stack-base AS stack-gazebo-build
ARG WS
WORKDIR ${WS}

# --skip-keys catkin: DecompUtil's package.xml says buildtool_depend catkin but
# it's a plain CMake project. Undeclared: Eigen (decomp_util) and boost_chrono
# (ros2_livox links it by name).
COPY --from=stack-sim-manifests /manifests /tmp/manifests
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths /tmp/manifests --skip-keys "catkin" \
 && apt-get install -y --no-install-recommends libeigen3-dev libboost-chrono-dev \
 && rm -rf /var/lib/apt/lists/* /tmp/manifests

# Same caps as deps-build (see JOBS above). Declared only now, below the
# installs, and exported only by the RUN lines that compile: an ARG is in the
# cache key of every RUN after it, so declaring it (or an ENV made from it)
# above the rosdep layer made another --jobs re-run the installs too.
ARG JOBS

# Livox-SDK2/livox_ros_driver2 aren't plain colcon packages (as in deps-build).
# They are copied and built on their own, ahead of the rest of the sources, so
# a pin bump of another sim repo does not rebuild them.
# Livox-SDK2: plain CMake project, cmake-installed to /usr/local (where the
# driver's CMakeLists looks for it).
COPY --from=stack-sim-src /Livox-SDK2 /opt/livox_sdk/src
RUN cd /opt/livox_sdk/src && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
 && make -C build -j"${JOBS}" && make -C build install \
 && ldconfig

# livox_ros_driver2: built via its own build.sh (picks the ROS 2 message
# variant) in its own workspace. build.sh exits 0 even when colcon fails, so
# check for the package's ament index entry, which is installed last.
COPY --from=stack-sim-src /livox_ros_driver2 /opt/livox_ws/src/livox_ros_driver2
RUN export MAKEFLAGS=-j${JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${JOBS} \
 && source /opt/ros/humble/setup.bash \
 && cd /opt/livox_ws/src/livox_ros_driver2 && ./build.sh humble \
 && test -f /opt/livox_ws/install/livox_ros_driver2/share/ament_index/resource_index/packages/livox_ros_driver2

# The colcon workspace: everything else in rover_sim.sim.repos (the two above
# are already built; colcon would take both for plain CMake packages).
# decomp_util first: decomp_rviz_plugins find_package()s it without declaring
# the dependency, so colcon can't order them in one invocation. livox_ws is
# sourced first so install/setup.bash chains it.
COPY --from=stack-sim-src / ${WS}/src
RUN rm -rf src/Livox-SDK2 src/livox_ros_driver2 \
 && export MAKEFLAGS=-j${JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${JOBS} \
 && source /opt/ros/humble/setup.bash && source /opt/livox_ws/install/setup.bash \
 && colcon build --executor sequential --packages-select decomp_util --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && source install/setup.bash \
 && colcon build --executor sequential --packages-skip decomp_util \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
 && rm -rf build log

# ---------- stack-rover-sim-build: this repo's rover_sim package -------------
# Its own stage so an edit to src/rover_sim rebuilds only this and the gazebo
# target's last COPY. On top of stack-gazebo-build for Gazebo's headers and
# gazebo_dev (the real_time_lock world plugin); only rover_sim is built here.
FROM stack-gazebo-build AS stack-rover-sim-build
ARG WS
WORKDIR ${WS}
COPY src/rover_sim ${WS}/src/rover_sim
RUN source /opt/ros/humble/setup.bash && source ${WS}/install/setup.bash \
 && colcon build --packages-select rover_sim \
 && rm -rf build log \
 && mkdir -p /manifests/rover_sim && cp src/rover_sim/package.xml /manifests/rover_sim/ \
 && find /manifests -exec touch -d @0 {} +

# ---------- gazebo: the sim-gazebo container ---------------------------------
FROM stack-base AS gazebo
ARG WS
LABEL rover_sim.stack="gazebo"
WORKDIR ${WS}
COPY --from=stack-sim-manifests /manifests /tmp/manifests
COPY --from=stack-rover-sim-build /manifests /tmp/manifests
# Exec deps from the manifests (Gazebo itself comes in via gazebo_dev), plus
# RViz, xacro, numpy for the sim's Python nodes, boost_chrono for ros2_livox,
# and stock Mesa (as in runtime). The last line fails this layer if any of it
# moved the rmw_zenoh pin.
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --dependency-types exec --from-paths /tmp/manifests --skip-keys "catkin" \
 && apt-get install -y --no-install-recommends \
        ros-humble-rviz2 ros-humble-xacro python3-numpy libboost-chrono1.74.0 \
        libgl1-mesa-glx libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers mesa-utils \
 && rm -rf /var/lib/apt/lists/* /tmp/manifests \
 && dpkg-query -W ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor | diff /etc/rover_sim/rmw-zenoh.pin -
# The Livox SDK's shared library (liblivox_ros_driver2.so links it) + the
# driver's install tree, then the workspace.
COPY --from=stack-gazebo-build /usr/local/lib/liblivox_lidar_sdk_shared.so /usr/local/lib/
RUN ldconfig
COPY --from=stack-gazebo-build /opt/livox_ws/install /opt/livox_ws/install
COPY --from=stack-gazebo-build ${WS}/install ${WS}/install
# Lets Gazebo resolve package://rover_sim/... meshes.
ENV GAZEBO_MODEL_PATH=${WS}/install/rover_sim/share
# Qt (RViz, gzclient) draws over the X11 socket; the container doesn't share
# the X server's IPC namespace, so MIT-SHM would fail.
ENV QT_X11_NO_MITSHM=1
# Read by the NVIDIA runtime only when compose gives this container a GPU;
# graphics adds the GL/EGL driver libraries Gazebo and RViz render with.
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
# Interactive shells (docker exec -it ... bash) get the panes' setup: Gazebo's
# own (model, resource and plugin paths; without it model://ground_plane does
# not resolve) and the workspace. install/setup.bash alone also chains livox_ws.
RUN printf 'source /opt/ros/humble/setup.bash\nsource /usr/share/gazebo/setup.bash\nsource /opt/livox_ws/install/setup.bash\nsource %s/install/setup.bash\n' "${WS}" >> /root/.bashrc
COPY --from=stack-rover-sim-build ${WS}/install/rover_sim ${WS}/install/rover_sim
# Fails the build if anything the sim's own nodes and launch files need at run
# time is missing: livox_sim_bridge.py and sim_tf.py (numpy, yaml, tf2_ros,
# Livox and std msgs), spawn_agent.launch.py (xacro, spawn_entity.py)
# and world.launch.py (gazebo_ros's init/factory plugins, rover_sim's
# real_time_lock, built only where Gazebo's headers are). All of it comes
# from the base image, rosdep, the apt line above or this workspace.
RUN source /opt/ros/humble/setup.bash && source ${WS}/install/setup.bash \
 && python3 -c "import numpy, yaml, rclpy, tf2_ros, std_srvs.srv, std_msgs.msg, nav_msgs.msg, sensor_msgs_py.point_cloud2, gazebo_msgs.srv, livox_ros_driver2.msg" \
 && command -v xacro >/dev/null && command -v gzserver >/dev/null \
 && test -x "$(ros2 pkg prefix gazebo_ros)/lib/gazebo_ros/spawn_entity.py" \
 && test -f "$(ros2 pkg prefix gazebo_ros)/lib/libgazebo_ros_factory.so" \
 && test -f "$(ros2 pkg prefix ros2_livox_simulation)/lib/libros2_livox.so" \
 && test -f "$(ros2 pkg prefix gazebo_plugins)/lib/libgazebo_ros_imu_sensor.so" \
 && test -f "$(ros2 pkg prefix rover_sim)/lib/librover_sim_real_time_lock.so"
# Command comes from compose: the container only idles (sleep infinity).

# ---------- stack-elevation-src: elevation_mapping_cupy ----------------------
FROM stack-base AS stack-elevation-git
ARG STACK_ELEVATION_GIT_URL
ARG STACK_ELEVATION_GIT_REF
RUN git clone "${STACK_ELEVATION_GIT_URL}" /src && git -C /src checkout "${STACK_ELEVATION_GIT_REF}"
FROM scratch AS stack-elevation-src
COPY --from=stack-elevation-git /src /

# ---------- stack-elevation-build: elevation_mapping_cupy + ros2_numpy -------
FROM stack-base AS stack-elevation-build
ARG WS
ARG STACK_ROS2_NUMPY_GIT_URL
ARG STACK_ROS2_NUMPY_GIT_REF
WORKDIR ${WS}
COPY --from=stack-elevation-src / ${WS}/src/elevation_mapping_cupy
# As in elevation-build: sensor_processing (semantic_sensor, for camera
# semantics) isn't used. plane_segmentation carries its own COLCON_IGNORE.
RUN git clone "${STACK_ROS2_NUMPY_GIT_URL}" src/ros2_numpy && git -C src/ros2_numpy checkout "${STACK_ROS2_NUMPY_GIT_REF}" \
 && touch src/elevation_mapping_cupy/sensor_processing/COLCON_IGNORE \
 && apt-get update && rosdep update \
 && rosdep install -y --ignore-src --from-paths src --skip-keys "semantic_sensor" \
 && rm -rf /var/lib/apt/lists/*
RUN source /opt/ros/humble/setup.bash \
 && colcon build --base-paths src --cmake-args -DBUILD_TESTING=OFF \
 && rm -rf build log
RUN mkdir -p /manifests && cd src \
 && find . -name package.xml -not -path "*/sensor_processing/*" -not -path "*/plane_segmentation/*" \
        -exec cp --parents {} /manifests \; \
 && find /manifests -exec touch -d @0 {} +

# ---------- elevation: the sim-elevation container ---------------------------
# Clean: stack-base + elevation only, nothing from the legacy runtime stage.
FROM stack-base AS elevation
ARG WS
ARG STACK_ELEVATION_GIT_REF
# The pinned ref. A --build-context build of another checkout still carries it
# unless overridden with --label rover_sim.elevation.ref=... (build_images.sh).
LABEL rover_sim.stack="elevation" \
      rover_sim.elevation.ref="${STACK_ELEVATION_GIT_REF}"
WORKDIR ${WS}
COPY --from=stack-elevation-build /manifests /tmp/manifests
RUN apt-get update && rosdep update \
 && rosdep install -y --ignore-src --dependency-types exec --from-paths /tmp/manifests --skip-keys "semantic_sensor" \
 && rm -rf /var/lib/apt/lists/* /tmp/manifests \
 && dpkg-query -W ros-humble-rmw-zenoh-cpp ros-humble-zenoh-cpp-vendor | diff /etc/rover_sim/rmw-zenoh.pin -
# The same Python pins as runtime-elevation. CUDA comes from pip (12.4, what
# l01's 550 driver supports): torch's cu124 wheels and CuPy's cuda-toolkit
# extra. numpy 2 (CuPy 14 needs it) breaks jammy's apt cv2, so OpenCV comes
# from pip too.
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0+cu124 \
        cupy-cuda12x==14.1.1 "cuda-toolkit[cublas,cudart,cufft,curand,cusolver,cusparse,nvrtc]==12.4.1" \
        opencv-python-headless==5.0.0.93 simple-parsing==0.1.9 shapely==2.1.2 \
        transforms3d==0.4.2 ruamel.yaml==0.19.1 numpy==2.2.6 scipy==1.15.3
# CuPy's cupyx.cusolver links libcusolver.so.11 by name, so the pip CUDA
# libraries have to be on the loader path. (The base image sets no
# LD_LIBRARY_PATH; the ROS setup files prepend theirs at run time.)
ARG NV=/usr/local/lib/python3.10/dist-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cusolver/lib:${NV}/cublas/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib
COPY --from=stack-elevation-build ${WS}/install ${WS}/install
RUN printf 'source /opt/ros/humble/setup.bash\nsource %s/install/setup.bash\n' "${WS}" >> /root/.bashrc
# Command comes from compose: the container only idles (sleep infinity).

# =============================================================================
# LEGACY MONOLITH: everything below builds rover-sim:local (make build). Left
# unchanged until Phase 6 removes it with docker/compose.yaml and launch/.
# =============================================================================

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
