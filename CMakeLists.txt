cmake_minimum_required(VERSION 2.8.3)
project(logo-recognition)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )

include_directories(
  include/
)

add_library(helper
  src/LogoRecognition.cpp
)

add_executable(logo-recognition
  src/main.cpp 
)

target_link_libraries(logo-recognition 
  ${OpenCV_LIBS} 
  helper
)