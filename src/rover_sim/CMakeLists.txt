cmake_minimum_required(VERSION 3.8)
project(rover_sim)

find_package(ament_cmake REQUIRED)
# Optional: the legacy image builds this package without Gazebo's headers and never loads
# the plugin. The stack's gazebo image builds it, and its build fails if the library is missing.
find_package(gazebo_dev QUIET)

if(gazebo_dev_FOUND)
  # World plugin that holds sim time to the wall clock (src/real_time_lock.cpp); the worlds load it.
  add_library(rover_sim_real_time_lock SHARED src/real_time_lock.cpp)
  target_compile_features(rover_sim_real_time_lock PRIVATE cxx_std_17)
  ament_target_dependencies(rover_sim_real_time_lock gazebo_dev)
  install(TARGETS rover_sim_real_time_lock LIBRARY DESTINATION lib)
else()
  message(STATUS "gazebo_dev not found: not building the real_time_lock world plugin")
endif()

# __pycache__: a launch file run from a bind-mounted checkout leaves one (root-owned) behind.
install(DIRECTORY config launch meshes urdf worlds DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE)

ament_package()
