add_executable(imm imm.cc)
target_link_libraries(imm
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

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

add_executable(mpi-hill-climbing mpi-hill-climbing.cc)
target_link_libraries(mpi-hill-climbing
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
endif()
