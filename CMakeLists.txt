cmake_minimum_required(VERSION 3.10)
project(SoccerBallField)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# → system libs
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# → Glad
add_library(glad STATIC
  ${CMAKE_SOURCE_DIR}/external/glad/src/glad.c
)
target_include_directories(glad PUBLIC
  ${CMAKE_SOURCE_DIR}/external/glad/include
  ${CMAKE_SOURCE_DIR}/external/stb
)

# → Dear ImGui
set(IMGUI_DIR   ${CMAKE_SOURCE_DIR}/external/imgui)
set(IMGUI_BACK  ${IMGUI_DIR}/backends)
add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_BACK}/imgui_impl_glfw.cpp
  ${IMGUI_BACK}/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
  ${IMGUI_DIR}
  ${IMGUI_BACK}
)
target_link_libraries(imgui PUBLIC glfw)

# → Your application
add_executable(soccer main.cpp)
target_include_directories(soccer PRIVATE
  ${CMAKE_SOURCE_DIR}/external/glad/include
  ${IMGUI_DIR}
  ${IMGUI_BACK}
  ${CMAKE_SOURCE_DIR}/external/glm
  ${CMAKE_SOURCE_DIR}/external/stb
)
target_link_libraries(soccer PRIVATE
  glad
  imgui
  glfw
  ${OPENGL_gl_LIBRARY}
)
