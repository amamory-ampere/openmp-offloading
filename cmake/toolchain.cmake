# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
set(CMAKE_CROSSCOMPILING "True")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
 
set(target_arch aarch64-linux-gnu)
set(CMAKE_LIBRARY_ARCHITECTURE ${target_arch} CACHE STRING "" FORCE)
 
set(CMAKE_SYSROOT "/home/aamory/mnt_xavier" CACHE STRING "" FORCE)
 
if (CMAKE_CROSSCOMPILING AND CMAKE_SYSROOT)
    # When cross compiling with CMake any calls to pkg_check_modules() will
    # search into the host system instead of the target sysroot.
    # The current work around is based on the discussion found at
    # https://gitlab.kitware.com/cmake/cmake/-/merge_requests/4478
    set(ENV{PKG_CONFIG_DIR} "")
    set(ENV{PKG_CONFIG_LIBDIR} "${CMAKE_SYSROOT}/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/pkgconfig")
    set(ENV{PKG_CONFIG_SYSROOT_DIR} ${CMAKE_SYSROOT})
endif() 

 
# Configure cmake to look for libraries, include directories and
# packages inside the target root prefix.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CMAKE_FIND_ROOT_PATH "~/tools/l4t-gcc/${target_arch}")
 
# needed to avoid doing some more strict compiler checks that
# are failing when cross-compiling
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
 
# specify the toolchain programs
find_program(CMAKE_C_COMPILER ${target_arch}-gcc)
find_program(CMAKE_CXX_COMPILER ${target_arch}-g++)
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "Can't find suitable C/C++ cross compiler for ${target_arch}")
endif()
 
set(CMAKE_AR ${target_arch}-ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ${target_arch}-ranlib)
set(CMAKE_LINKER ${target_arch}-ld)

# flags for the Xavier
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.2-a" CACHE INTERNAL "" FORCE)
set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -march=armv8.2-a" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -march=armv8.2-a" CACHE INTERNAL "" FORCE)

# set in includes
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${CMAKE_SYSROOT}/usr/include" CACHE INTERNAL "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_SYSROOT}/usr/include" CACHE INTERNAL "" FORCE)


# Not all shared libraries dependencies are instaled in host machine.
# Make sure linker doesn't complain.
set(CMAKE_EXE_LINKER_FLAGS_INIT -Wl,--allow-shlib-undefined)
 
# instruct nvcc to use our cross-compiler
set(CMAKE_CUDA_FLAGS "-ccbin ${CMAKE_CXX_COMPILER} -Xcompiler -fPIC" CACHE STRING "" FORCE)
