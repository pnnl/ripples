add_executable(imm imm.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(imm.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(imm CUDA::cudart)
endif()
target_link_libraries(imm
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      infmax)
if (metall_FOUND)
  target_compile_definitions(imm PRIVATE ENABLE_METAL=1)
  target_link_libraries(imm metall::metall)
endif()

target_compile_definitions(imm PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>)


add_executable(simulator simulator.cc)
target_link_libraries(simulator
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

add_executable(dump-graph dump-graph.cc)
target_link_libraries(dump-graph
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

add_executable(hill-climbing hill_climbing.cc)
target_link_libraries(hill-climbing
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

if (MPI_FOUND)
add_executable(mpi-imm mpi-imm.cc)
target_link_libraries(mpi-imm
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
if (metall_FOUND)
  target_compile_definitions(mpi-imm PRIVATE ENABLE_METAL=1)
  target_link_libraries(mpi-imm metall::metall)
endif()

add_executable(mpi-hill-climbing mpi-hill-climbing.cc)
target_link_libraries(mpi-hill-climbing
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
endif()
