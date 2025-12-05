cmake_minimum_required(VERSION 3.0.2)
project(hookload_det)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        #pcl_ros
        #pcl_conversions
)

# 设置PCL的自定义安装路径
set(PCL_DIR "/usr/local/share/pcl-1.12")
find_package(PCL 1.12 REQUIRED)
find_package(PCL  REQUIRED )


find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("found openMP!!!")
    add_compile_definitions("USE_OPENMP")
endif()
#add_compile_definitions("CSF_DATA_TYPE")
#add_compile_definitions("VIS_DEBUG_MODE")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES livox_avia_data
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
set(CSF_SOURCES
    src/CSF/c2cdist.cpp
    src/CSF/Cloth.cpp
    src/CSF/Constraint.cpp
    src/CSF/CSF.cpp
    src/CSF/Particle.cpp
    src/CSF/point_cloud.cpp
    src/CSF/Rasterization.cpp
    src/CSF/XYZReader.cpp
)
set(CSF_HEADERS
    include/CSF/c2cdist.h
    include/CSF/Cloth.h
    include/CSF/Constraint.h
    include/CSF/CSF.h
    include/CSF/Particle.h
    include/CSF/point_cloud.h
    include/CSF/Rasterization.h
    include/CSF/Vec3.h
    include/CSF/XYZReader.h
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# 链接PCL的库文件目录
link_directories(${PCL_LIBRARY_DIRS})


add_executable(hook_det_and_track_mouseclick_test  
                src/livox_test.cpp 
                src/HookLoadPositionAcquirer.cpp 
                src/ICPTracker.cpp
                src/CollisionDetector.cpp
                ${CSF_SOURCES}
# src/CounterweightCrack.cpp
)

target_link_libraries(  hook_det_and_track_mouseclick_test 
                        ${catkin_LIBRARIES}
                        ${PCL_LIBRARIES}
                        OpenMP::OpenMP_CXX)


