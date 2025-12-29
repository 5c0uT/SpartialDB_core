cmake_minimum_required(VERSION 3.25)
project(SpatialDB_Core LANGUAGES CXX C)

set(CMAKE_MESSAGE_LOG_LEVEL VERBOSE)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LOG_FILE "${CMAKE_BINARY_DIR}/cmake_detailed.log")
file(WRITE "${LOG_FILE}" "=== SpatialDB_Core CMake Build System ===\n")

message(STATUS "")
message(STATUS "========== SpatialDB_Core CMake Build System ==========")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Binary Dir: ${CMAKE_BINARY_DIR}")
message(STATUS "Source Dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "========================================================")
message(STATUS "")

file(APPEND "${LOG_FILE}" "CMake Version: ${CMAKE_VERSION}\n")
file(APPEND "${LOG_FILE}" "System: ${CMAKE_SYSTEM_NAME}\n\n")

if(MSVC)
    message(STATUS "Compiler: MSVC (Visual Studio)")
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    
    string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "/MT" "/MD" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "/MT" "/MD" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/MT" "/MD" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "/MTd" "/MDd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/MTd" "/MDd" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    
    set(CMAKE_CXX_FLAGS_RELEASE "/DWIN32 /D_WINDOWS /GR /EHsc /MP /MD /O2 /Ob2 /D NDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "/DWIN32 /D_WINDOWS /MP /MD /O2 /D NDEBUG")
    
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " /ignore:4098")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS " /ignore:4098")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /ignore:4098")
    
    message(STATUS "Runtime: MultiThreadedDLL (/MD)")
    file(APPEND "${LOG_FILE}" "Compiler: MSVC\n")
    file(APPEND "${LOG_FILE}" "Runtime: /MD (MultiThreadedDLL)\n\n")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Compiler: Clang")
    file(APPEND "${LOG_FILE}" "Compiler: Clang\n\n")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(STATUS "Compiler: GCC")
    file(APPEND "${LOG_FILE}" "Compiler: GCC\n\n")
else()
    message(WARNING "Unknown compiler")
    file(APPEND "${LOG_FILE}" "Compiler: Unknown\n\n")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
file(APPEND "${LOG_FILE}" "Build Type: ${CMAKE_BUILD_TYPE}\n")

set(LOCAL_DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies" CACHE PATH "Dependencies")
set(PYTHON_ROOT "C:\\Python314" CACHE PATH "Python root")
set(CONDA_ENV_PATH "C:\\ProgramData\\envs\\spatial_env" CACHE PATH "Conda environment")

message(STATUS "")
message(STATUS "Paths:")
message(STATUS "  Dependencies: ${LOCAL_DEPS_DIR}")
message(STATUS "  Python: ${PYTHON_ROOT}")
message(STATUS "  Conda: ${CONDA_ENV_PATH}")
message(STATUS "")

file(APPEND "${LOG_FILE}" "\nConfiguration Paths:\n")
file(APPEND "${LOG_FILE}" "  Dependencies: ${LOCAL_DEPS_DIR}\n")
file(APPEND "${LOG_FILE}" "  Python: ${PYTHON_ROOT}\n")
file(APPEND "${LOG_FILE}" "  Conda: ${CONDA_ENV_PATH}\n\n")

message(STATUS "Searching for pybind11...")
if(pybind11_FOUND)
    message(STATUS "pybind11: FOUND")
else()
    message(STATUS "pybind11: NOT FOUND - Fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.1
    )
    FetchContent_MakeAvailable(pybind11)
    message(STATUS "pybind11: CONFIGURED")
endif()

message(STATUS "Searching for Python...")
set(Python_ROOT_DIR "${PYTHON_ROOT}")
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(Python_FOUND)
    message(STATUS "Python: FOUND (${Python_VERSION})")
    file(APPEND "${LOG_FILE}" "Python: FOUND (${Python_VERSION})\n")
else()
    message(FATAL_ERROR "Python NOT FOUND")
endif()

message(STATUS "Searching for PROJ...")
set(PROJ_DIR "${CONDA_ENV_PATH}/Library")
find_package(PROJ PATHS "${PROJ_DIR}" CONFIG QUIET)

if(PROJ_FOUND)
    message(STATUS "PROJ: FOUND")
    set(USE_PROJ TRUE)
    file(APPEND "${LOG_FILE}" "PROJ: FOUND\n")
else()
    message(STATUS "PROJ: NOT FOUND")
    set(USE_PROJ FALSE)
    file(APPEND "${LOG_FILE}" "PROJ: NOT FOUND\n")
endif()

message(STATUS "Searching for CURL...")
set(CURL_DIR "${CONDA_ENV_PATH}/Library")
find_package(CURL PATHS "${CURL_DIR}" CONFIG QUIET)

if(CURL_FOUND)
    message(STATUS "CURL: FOUND")
    set(USE_CURL TRUE)
    file(APPEND "${LOG_FILE}" "CURL: FOUND\n")
else()
    message(STATUS "CURL: NOT FOUND")
    set(USE_CURL FALSE)
    file(APPEND "${LOG_FILE}" "CURL: NOT FOUND\n")
endif()

message(STATUS "Searching for PhysX...")
set(VCPKG_BASE "${LOCAL_DEPS_DIR}/vcpkg/installed/x64-windows")
set(VCPKG_INCLUDE_CHECK "${VCPKG_BASE}/include/physx")

if(EXISTS "${VCPKG_BASE}" AND EXISTS "${VCPKG_INCLUDE_CHECK}")
    message(STATUS "PhysX: FOUND")
    
    set(PHYSX_INCLUDE_DIR "${VCPKG_BASE}/include")
    set(PHYSX_LIB_DIR "${VCPKG_BASE}/lib")
    
    file(GLOB PHYSX_LIBS_LOWER "${PHYSX_LIB_DIR}/*physx*.lib")
    file(GLOB PHYSX_LIBS_UPPER "${PHYSX_LIB_DIR}/*PhysX*.lib")
    file(GLOB PHYSX_LIBS_MIXED "${PHYSX_LIB_DIR}/*[Pp]hysX*.lib")
    
    if(PHYSX_LIBS_LOWER)
        set(PHYSX_LIBRARIES "${PHYSX_LIBS_LOWER}")
    elseif(PHYSX_LIBS_UPPER)
        set(PHYSX_LIBRARIES "${PHYSX_LIBS_UPPER}")
    elseif(PHYSX_LIBS_MIXED)
        set(PHYSX_LIBRARIES "${PHYSX_LIBS_MIXED}")
    endif()
    
    if(PHYSX_LIBRARIES)
        list(LENGTH PHYSX_LIBRARIES PHYSX_LIB_COUNT)
        message(STATUS "PhysX Libraries: ${PHYSX_LIB_COUNT} files found")
        file(APPEND "${LOG_FILE}" "PhysX: FOUND (${PHYSX_LIB_COUNT} libraries)\n")
        
        foreach(lib ${PHYSX_LIBRARIES})
            get_filename_component(lib_name "${lib}" NAME)
            message(STATUS "  - ${lib_name}")
        endforeach()
        
        set(PHYSX_FOUND TRUE)
    else()
        message(FATAL_ERROR "PhysX libraries not found")
    endif()
else()
    message(FATAL_ERROR "PhysX not found at ${VCPKG_BASE}")
    set(PHYSX_FOUND FALSE)
endif()

message(STATUS "")
message(STATUS "Collecting source files...")

set(CORE_SOURCES
    src/pybind_module.cpp
    src/SpatialDB.cpp
)

set(PHYSX_SOURCES "")
if(PHYSX_FOUND)
    set(PHYSX_SOURCES
        src/PhysXCore.cpp
        src/Voxelizer.cpp
        src/BVHManager.cpp
    )
    message(STATUS "  Adding PhysX sources...")
endif()

set(PROJ_SOURCES "")
if(USE_PROJ)
    set(PROJ_SOURCES src/CoordinateConverter.cpp)
    message(STATUS "  Adding PROJ sources...")
endif()

set(FINAL_SOURCES ${CORE_SOURCES} ${PHYSX_SOURCES} ${PROJ_SOURCES})

if(PHYSX_FOUND)
    message(STATUS "Creating static library: spatialdb_core")
    
    add_library(spatialdb_core STATIC ${FINAL_SOURCES})
    
    if(MSVC)
        set_target_properties(spatialdb_core PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
        target_compile_options(spatialdb_core PRIVATE /W4 /wd4100)
        target_compile_definitions(spatialdb_core PRIVATE NDEBUG)
    endif()
    
    target_include_directories(spatialdb_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PHYSX_INCLUDE_DIR}"
        "${PHYSX_INCLUDE_DIR}/physx"
        "${Python_INCLUDE_DIRS}"
        "${pybind11_INCLUDE_DIR}"
    )
    
    target_link_libraries(spatialdb_core PUBLIC
        ${PHYSX_LIBRARIES}
        ws2_32
        winmm
    )
    
    message(STATUS "Library spatialdb_core: CONFIGURED")
else()
    message(STATUS "Creating minimal library (no PhysX)")
    add_library(spatialdb_core STATIC ${CORE_SOURCES})
    
    if(MSVC)
        set_target_properties(spatialdb_core PROPERTIES
            MSVC_RUNTIME_LIBRARY MultiThreadedDLL
        )
        target_compile_options(spatialdb_core PRIVATE /W4 /wd4100)
        target_compile_definitions(spatialdb_core PRIVATE NDEBUG)
    endif()
    
    target_include_directories(spatialdb_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${Python_INCLUDE_DIRS}"
        "${pybind11_INCLUDE_DIR}"
    )
    
    target_link_libraries(spatialdb_core PUBLIC ws2_32 winmm)
endif()

message(STATUS "Creating Python module: spatialdb_core_pybind")

pybind11_add_module(spatialdb_core_pybind
    SHARED
    src/pybind_module.cpp
)

if(MSVC)
    set_target_properties(spatialdb_core_pybind PROPERTIES
        MSVC_RUNTIME_LIBRARY MultiThreadedDLL
    )
    target_compile_options(spatialdb_core_pybind PRIVATE /W4 /wd4100)
    target_compile_definitions(spatialdb_core_pybind PRIVATE NDEBUG)
endif()

target_include_directories(spatialdb_core_pybind BEFORE PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${PHYSX_INCLUDE_DIR}"
    "${PHYSX_INCLUDE_DIR}/physx"
    "${Python_INCLUDE_DIRS}"
    "${pybind11_INCLUDE_DIR}"
)

target_link_libraries(spatialdb_core_pybind PRIVATE
    spatialdb_core
    ${Python_LIBRARIES}
)

if(PHYSX_FOUND)
    target_link_libraries(spatialdb_core_pybind PRIVATE
        ${PHYSX_LIBRARIES}
        ws2_32
        winmm
    )
endif()

set_target_properties(spatialdb_core_pybind PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PREFIX ""
)

message(STATUS "Module spatialdb_core_pybind: CONFIGURED")

message(STATUS "")
message(STATUS "========== BUILD CONFIGURATION SUMMARY ==========")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "CMake: ${CMAKE_VERSION}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Python: YES")
message(STATUS "  pybind11: YES")
message(STATUS "  PhysX: ${PHYSX_FOUND}")
message(STATUS "  PROJ: ${USE_PROJ}")
message(STATUS "  CURL: ${USE_CURL}")
message(STATUS "")
message(STATUS "Build Targets:")
message(STATUS "  1. spatialdb_core (STATIC)")
message(STATUS "  2. spatialdb_core_pybind (PYTHON)")
message(STATUS "")
message(STATUS "Detailed log: ${LOG_FILE}")
message(STATUS "================================================")
message(STATUS "")