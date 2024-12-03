include (ExternalProject)

set(protobuf-c_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/include)
set(protobuf-c_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/lib/libprotobuf-c.a)
set(protobuf-c_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/bin/protoc-c)

if(cuBERT_SYSTEM_PROTOBUF)
    find_package(Protobuf 3.0.0 REQUIRED)
else()
    include(protobuf)
    set(protobuf-c_DEPENDS protobuf)
    set(protobuf-c_PKG_CONFIG "PKG_CONFIG_PATH=${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/pkgconfig")
endif()

ExternalProject_Add(protobuf-c
    PREFIX protobuf-c
    DEPENDS ${protobuf-c_DEPENDS}
    GIT_REPOSITORY https://github.com/protobuf-c/protobuf-c.git
    GIT_TAG v1.3.2
    GIT_SHALLOW 1
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/src/protobuf-c
    CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/src/protobuf-c/autogen.sh
    BUILD_COMMAND ${protobuf-c_PKG_CONFIG} CFLAGS=-fPIC CXXFLAGS=-fPIC
    ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/src/protobuf-c/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/protobuf-c
    INSTALL_COMMAND make install
)
