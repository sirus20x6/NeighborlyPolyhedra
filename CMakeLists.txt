cmake_minimum_required(VERSION 3.14)
project(NeighborlyPolyhedra)

# Set C++ standard to 20 (required for <format>).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
set(SOURCES
    main.cpp
    solver.cpp
    util.cpp
    simd_util.cpp
)

set(HEADERS
    plane.h
    solver.h
    util.h
    simd_util.h
    cpu_features.h
    memory_pool.h
)

# Find dependencies
find_package(Eigen3 3.4 REQUIRED)

# Cairo is optional
option(USE_CAIRO "Enable Cairo support for rendering cutouts" OFF)
if(USE_CAIRO)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(CAIRO cairo)
    endif()
    
    if(NOT CAIRO_FOUND)
        find_package(Cairo)
    endif()
    
    if(CAIRO_FOUND)
        add_definitions(-DUSE_CAIRO)
    else()
        message(WARNING "Cairo not found. Cutout rendering will be disabled.")
    endif()
endif()

# ------------------------------------------------------------------------------
# Always include termcolor, no toggle needed
# ------------------------------------------------------------------------------
# 1) Put termcolor.hpp in, say, external/termcolor/termcolor.hpp
# 2) Add that path to your includes:
add_definitions(-DUSE_TERMCOLOR)  # If you want a #define in your code
                                  # otherwise you can omit this line.

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Always add termcolor's include directory
target_include_directories(${PROJECT_NAME} PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/external/termcolor"   # Adjust if needed
)

# Compiler-specific flags for SIMD support
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Release>:/O2 /arch:AVX2>
        $<$<CONFIG:Debug>:/Od>
    )
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        -msse4.1 -mavx2  # Add SSE4.1 support along with AVX2
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
)

# Add Cairo if enabled and found
if(USE_CAIRO AND CAIRO_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${CAIRO_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CAIRO_LIBRARIES})
endif()

# Install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

# Install auxiliary files
install(FILES
    topologies.txt
    README.md
    LICENSE
    DESTINATION share/${PROJECT_NAME}
)
