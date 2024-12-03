# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
include (ExternalProject)

# NOTE: Different from mkldnn.cmake, this file is meant to download mkl libraries
set(MKL_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/include)
set(MKL_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/mkl/bin)
set(mkl_WIN mklml_win_2019.0.3.20190220.zip)
set(mkl_MAC mklml_mac_2019.0.3.20190220.tgz)
set(mkl_LNX mklml_lnx_2019.0.3.20190220.tgz)
set(mkl_TAG v0.18)
set(mkl_URL https://github.com/intel/mkl-dnn/releases)

if (WIN32)
    set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_WIN})
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/mklml.dll)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5md.dll)
elseif (UNIX AND NOT APPLE)
    set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_LNX})
    set(mkl_MD5 76354b74325cd293aba593d7cbe36b3f)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5.so)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libmklml_gnu.so)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libmklml_intel.so)
elseif (APPLE)
    set(mkl_DOWNLOAD_URL ${mkl_URL}/download/${mkl_TAG}/${mkl_MAC})
    set(mkl_MD5 3b28da686a25a4cf995ca4fc5e30e514)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libiomp5.dylib)
    list(APPEND MKL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/mkl/src/mkl/lib/libmklml.dylib)
endif ()

ExternalProject_Add(mkl
        PREFIX mkl
        URL ${mkl_DOWNLOAD_URL}
        URL_MD5 ${mkl_MD5}
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND "")

# put mkl dynamic libraries in one bin directory
add_custom_target(mkl_create_destination_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${MKL_BIN_DIR}
        DEPENDS mkl)

add_custom_target(mkl_copy_shared_to_destination DEPENDS mkl_create_destination_dir)

foreach(dll_file ${MKL_LIBRARIES})
    add_custom_command(TARGET mkl_copy_shared_to_destination PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll_file} ${MKL_BIN_DIR})
endforeach()
