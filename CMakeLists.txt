cmake_minimum_required(VERSION 3.15)
project(SpatialDBCore)

# Установка единого режима линковки
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Настройки путей вывода
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    add_definitions(-D_USE_MATH_DEFINES)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Пути к зависимостям
set(CONDA_ENV_PATH "C:/ProgramData/miniforge3/envs/spatial_env")
set(PHYSX_ROOT "C:/PhysX-107.0-physx-5.6.0/physx")
set(PHYSX_INCLUDE_DIR "${PHYSX_ROOT}/include")
set(PHYSX_LIB_DIR "${PHYSX_ROOT}/bin/win.x86_64.vc143.mt/release")

# Используем СТАТИЧЕСКИЕ библиотеки PhysX (обратите внимание на суффиксы)
set(PHYSX_LIBRARIES
    "${PHYSX_LIB_DIR}/PhysX_64.lib"
    "${PHYSX_LIB_DIR}/PhysXFoundation_64.lib"
    "${PHYSX_LIB_DIR}/PhysXCommon_64.lib"
    "${PHYSX_LIB_DIR}/PhysXPvdSDK_static_64.lib"
    "${PHYSX_LIB_DIR}/PhysXExtensions_static_64.lib"
)

# Другие зависимости
set(CURL_INCLUDE_DIR "${CONDA_ENV_PATH}/Library/include")
set(CURL_LIBRARY "${CONDA_ENV_PATH}/Library/lib/libcurl.lib")
set(PROJ_INCLUDE_DIR "${CONDA_ENV_PATH}/Library/include")
set(PROJ_LIBRARY "${CONDA_ENV_PATH}/Library/lib/proj.lib")

# Поиск Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PHYSX_INCLUDE_DIR}
    ${PROJ_INCLUDE_DIR}
    ${CURL_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

set(SOURCES
    src/BVHManager.cpp
    src/PhysXCore.cpp
    src/CoordinateConverter.cpp
    src/SpatialDB.cpp
    src/pybind_module.cpp
    src/Voxelizer.cpp
)

# Основная библиотека
add_library(spatialdb_core STATIC ${SOURCES})
set_target_properties(spatialdb_core PROPERTIES
    DEBUG_POSTFIX ""
    RELEASE_POSTFIX ""
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
target_link_libraries(spatialdb_core PRIVATE
    ${PROJ_LIBRARY}
    ${CURL_LIBRARY}
    ${PHYSX_LIBRARIES}
    ws2_32.lib
    winmm.lib
)

# Python модуль
add_library(spatialdb_core_pybind MODULE src/pybind_module.cpp)
set_target_properties(spatialdb_core_pybind PROPERTIES
    OUTPUT_NAME "spatialdb_core_pybind"
    SUFFIX ".pyd"
    DEBUG_POSTFIX ""
    RELEASE_POSTFIX ""
    PREFIX ""
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

target_link_options(spatialdb_core_pybind PRIVATE "/ignore:4098")
target_link_libraries(spatialdb_core_pybind PRIVATE
    spatialdb_core
    ${PROJ_LIBRARY}
    ${CURL_LIBRARY}
    ${PHYSX_LIBRARIES}
    ws2_32.lib
    winmm.lib
    ${Python_LIBRARIES}
)

# Копирование зависимостей
if(WIN32)
    add_custom_command(TARGET spatialdb_core_pybind POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:spatialdb_core_pybind>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PHYSX_LIB_DIR}/PhysXCommon_64.dll"
            $<TARGET_FILE_DIR:spatialdb_core_pybind>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PHYSX_LIB_DIR}/PhysXFoundation_64.dll"
            $<TARGET_FILE_DIR:spatialdb_core_pybind>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CONDA_ENV_PATH}/Library/bin/libcurl.dll"
            $<TARGET_FILE_DIR:spatialdb_core_pybind>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CONDA_ENV_PATH}/Library/bin/proj_9_2.dll"
            "$<TARGET_FILE_DIR:spatialdb_core_pybind>/proj.dll"
        COMMENT "Copying dependencies..."
    )
endif()

message(STATUS "PhysX lib path: ${PHYSX_LIB_DIR}")
message(STATUS "Libraries: ${PHYSX_LIBRARIES}")