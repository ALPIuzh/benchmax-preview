cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

project(benchmax LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost 1.81 REQUIRED COMPONENTS headers)
find_package(OpenMP REQUIRED)

pybind11_add_module(
  _exp_a_spiel
  MODULE
  NO_EXTRAS
  src/benchmax/integrations/exp_a_spiel/_exp_a_spiel.cpp
  src/benchmax/integrations/exp_a_spiel/_liars_dice.cpp
  third_party/exp_a_spiel/csrc/traverser.cpp
)
set_target_properties(
  _exp_a_spiel
  PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_include_directories(
  _exp_a_spiel
  PRIVATE
  third_party/exp_a_spiel/csrc
)
target_link_libraries(
  _exp_a_spiel
  PRIVATE
  Boost::headers
  OpenMP::OpenMP_CXX
)

install(
  TARGETS _exp_a_spiel
  LIBRARY DESTINATION benchmax/integrations/exp_a_spiel
  RUNTIME DESTINATION benchmax/integrations/exp_a_spiel
)
install(
  FILES
  third_party/exp_a_spiel/LICENSE
  third_party/exp_a_spiel/UPSTREAM.md
  DESTINATION benchmax/integrations/exp_a_spiel/third_party
)
