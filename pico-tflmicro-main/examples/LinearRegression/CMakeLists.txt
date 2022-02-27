
cmake_minimum_required(VERSION 3.12)

project(LinearRegression C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


add_executable(LinearRegression "")

target_include_directories(LinearRegression
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/.
)

set_target_properties(
  LinearRegression
  PROPERTIES
  COMPILE_FLAGS -fno-rtti
  COMPILE_FLAGS -fno-exceptions
  COMPILE_FLAGS -fno-threadsafe-statics
  COMPILE_FLAGS -nostdlib
)

target_sources(LinearRegression
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/main.cpp
  ${CMAKE_CURRENT_LIST_DIR}/tensorflow/model_data.h
)

target_link_libraries(
  LinearRegression
  pico-tflmicro
  hardware_pwm
)

pico_add_extra_outputs(LinearRegression)

# enable usb output, disable uart output // this is needed when we need to print someting
# minicom -b 115200 -o -D /dev/ttyACM0

pico_enable_stdio_usb(LinearRegression 1)
pico_enable_stdio_uart(LinearRegression 0)

