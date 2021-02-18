cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(signalnet)

include_directories( ./include ./src )

set(CMAKE_PREFIX_PATH "D:/code/libtorch")
set(BOOST_ROOT "D:/Boost")

find_package( Torch REQUIRED )
find_package( Boost 1.75.0 REQUIRED COMPONENTS filesystem)
message(STATUS "Boost version: ${Boost_VERSION}") 

add_executable(${PROJECT_NAME} src/main.cpp)

if(Boost_FOUND)
     include_directories(${Boost_INCLUDE_DIR})
     message(STATUS "Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}") 
     target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS}) 
     message(STATUS "Boost_LIBRARIES  " ${Boost_LIBRARIES}) 
    #target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME}  ${TORCH_LIBRARIES} ${Boost_LIBRARIES})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET signalnet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:signalnet>)
endif (MSVC)