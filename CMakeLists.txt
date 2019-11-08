cmake_minimum_required(VERSION 2.8)

project (test_caffe2_cpp)

set(CMAKE_MODULE_PATH /home/zf/workspaces/workspace_cpp/pytorch/cmake/Modules)

add_definitions(-D_DEBUG)
add_definitions(-DDEBUG)
add_definitions("$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

set(CAFFE2_SRC_PATH /home/zf/workspaces/workspace_cpp/pytorch/)
set(CAFFE2_SRC_INC_PATH ${CAFFE2_SRC_PATH}/torch/include/)

set(CAFFE2_BLD_PATH /home/zf/workspaces/workspace_cpp/pytorch/build/)
set(CAFFE2_GEN_INC_PATH ${CAFFE2_BLD_PATH}/lib.linux-x86_64-2.7/torch/include/)
set(CAFFE2_GEN_LIB_PATH ${CAFFE2_BLD_PATH}/lib/)
set(CAFFE2_LIB_PATH /home/zf/workspaces/workspace_cpp/pytorch/build/lib.linux-x86_64-2.7/torch/lib/)
find_package(Protobuf REQUIRED)
#find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cURL QUIET)
find_package(CUDA QUIET)
find_package(Threads REQUIRED)

include_directories(include)
include_directories(${EIGEN3_INCLUDE_DIR})
if(OpenCV_LIBS)
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()
############################### Mine

include_directories(${CAFFE2_SRC_INC_PATH})
include_directories(${CAFFE2_GEN_INC_PATH})

link_directories(${CAFFE2_LIB_PATH})

find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)
find_library(GTEST_LIB gtest)

add_subdirectory( src/util )

set(STATIC_LIB_PATH ${CAFFE2_GEN_LIB_PATH})
set(static_libs ${STATIC_LIB_PATH}/libprotoc.a 
	${STATIC_LIB_PATH}/libprotobuf.a 
	${STATIC_LIB_PATH}/libpthreadpool.a
	${STATIC_LIB_PATH}/libc10d.a)

set (shared_libs opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc torch glog gflags boost_system c10 rt pthread leveldb lmdb)


add_executable(intro src/intro.cpp)
target_link_libraries( intro ${shared_libs} ${static_libs})    

add_executable(toy src/toy.cpp)
target_link_libraries(toy ${shared_libs} ${static_libs})    

add_executable(mnist src/mnist.cpp $<TARGET_OBJECTS:util>)
target_link_libraries(mnist ${shared_libs} ${static_libs})

add_executable(rnn src/rnn.cpp $<TARGET_OBJECTS:util> )
target_link_libraries(rnn ${shared_libs} ${static_libs})

add_executable(testdb src/testdb.cpp)
target_link_libraries(testdb ${shared_libs} ${static_libs})    
