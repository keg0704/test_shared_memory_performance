cmake_minimum_required(VERSION 3.25)

project(multithreaded_decoding LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)
option(DOCKER "" ON)

# install prerequisites to system lib in docker environment
if(DOCKER)
  set(EXTERNAL_INSTALL_LOCATION /usr/local CACHE INTERNAL "")
else(DOCKER)
  set(EXTERNAL_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external CACHE INTERNAL "")
endif(DOCKER)

# install under the build directory outside docker
set(EXTERNAL_INCLUDE ${EXTERNAL_INSTALL_LOCATION}/include CACHE INTERNAL "")
option(TRITONBACKEND "" OFF)

ExternalProject_Add(opencv_build
   GIT_REPOSITORY https://github.com/opencv/opencv
    GIT_TAG 4.6.0
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION} -WITH_JPEG=ON
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j${NCPU} install
    )

ExternalProject_Add(libjpeg-turbo
    GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
    GIT_TAG main
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j${NCPU} install
    )

set(OPENCV_CORE_LIB ${EXTERNAL_INSTALL_LOCATION}/lib/libopencv_core.so CACHE INTERNAL "")
set(OPENCV_IMGPROC_LIB ${EXTERNAL_INSTALL_LOCATION}/lib/libopencv_imgproc.so CACHE INTERNAL "")
set(OPENCV_IMGCODEC_LIB ${EXTERNAL_INSTALL_LOCATION}/lib/libopencv_imgcodecs.so CACHE INTERNAL "")
set(TURBOJPEG_LIB ${EXTERNAL_INSTALL_LOCATION}/lib/libturbojpeg.so CACHE INTERNAL "")

### tjbenchmark library
file(GLOB gip "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

add_executable(tjbenchmark ${gip})
target_link_libraries(tjbenchmark PRIVATE stdc++fs ${OPENCV_CORE_LIB} ${OPENCV_IMGPROC_LIB} ${OPENCV_IMGCODEC_LIB} ${TURBOJPEG_LIB})
target_include_directories(tjbenchmark PRIVATE include ${EXTERNAL_INCLUDE} ${EXTERNAL_INCLUDE}/opencv4)

target_compile_options(tjbenchmark PRIVATE -fPIC)

