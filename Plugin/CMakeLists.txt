# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.2)
project(NvPipe VERSION 1.0.0 LANGUAGES CXX)

SET(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
        STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(CUDA REQUIRED)

# Construct path to CUDA driver API lib (not provided by FindCUDA)
get_filename_component(CUDA_LIB_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Options
option(NVPIPE_WITH_ENCODER "Enables the NvPipe encoding interface." ON)
option(NVPIPE_WITH_DECODER "Enables the NvPipe decoding interface." ON)
option(NVPIPE_WITH_OPENGL "Enables the NvPipe OpenGL interface." ON)
option(NVPIPE_BUILD_EXAMPLES "Builds the NvPipe example applications (requires both encoder and decoder)." ON)

# Header
configure_file(src/NvPipe.h.in include/NvPipe.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(./src)

# NvPipe shared library
list(APPEND NVPIPE_SOURCES
    src/NvPipe.cu
    src/Video_Codec_SDK_9.0.20/Samples/Utils/ColorSpace.cu
    )
list(APPEND NVPIPE_LIBRARIES
    ${CMAKE_DL_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_LIB}
    )

if (NVPIPE_WITH_ENCODER)
    list(APPEND NVPIPE_SOURCES
        src/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvEncoder/NvEncoder.cpp
        src/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvEncoder/NvEncoderCuda.cpp
        )
endif()

if (NVPIPE_WITH_DECODER)
    list(APPEND NVPIPE_SOURCES
        src/Video_Codec_SDK_9.0.20/Samples/NvCodec/NvDecoder/NvDecoder.cpp
        )
    list(APPEND NVPIPE_LIBRARIES
        nvcuvid
        )

    if (WIN32)
        if (CMAKE_SIZEOF_VOID_P EQUAL 8)
            link_directories(src/Video_Codec_SDK_9.0.20/Lib/x64)
        elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
            link_directories(src/Video_Codec_SDK_9.0.20/Lib/Win32)
        endif()
    endif()
endif()

include(GNUInstallDirs)

cuda_add_library(${PROJECT_NAME} SHARED ${NVPIPE_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
    )
target_include_directories(${PROJECT_NAME} PRIVATE
    $<BUILD_INTERFACE:src/Video_Codec_SDK_9.0.20/Samples src/Video_Codec_SDK_9.0.20/Samples/NvCodec src/Video_Codec_SDK_9.0.20/include ${CUDA_INCLUDE_DIRS}>
    )
target_include_directories(${PROJECT_NAME} PRIVATE src/Unity)
target_link_libraries(${PROJECT_NAME} ${NVPIPE_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)

#Use staticlly linked CRT for windows, because I use VS2019, if this is not set, the dll won't be able to run on a earlier version windows.
if (WIN32)
set_property(TARGET ${PROJECT_NAME} PROPERTY
  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

export(TARGETS ${PROJECT_NAME} FILE NvPipeConfig.cmake)
