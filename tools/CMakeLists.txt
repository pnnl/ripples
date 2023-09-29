add_executable(imm imm.cc)
target_link_libraries(imm
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX)

add_executable(simulator simulator.cc)
target_link_libraries(simulator
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX)

add_executable(dump-graph dump-graph.cc)
target_link_libraries(dump-graph
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX)

add_executable(hill-climbing hill_climbing.cc)
target_link_libraries(hill-climbing
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX)

if (MPI_FOUND)
add_executable(mpi-imm mpi-imm.cc)
target_link_libraries(mpi-imm
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)

add_executable(mpi-hill-climbing mpi-hill-climbing.cc)
target_link_libraries(mpi-hill-climbing
                      CONAN_PKG::spdlog CONAN_PKG::fmt CONAN_PKG::libtrng
                      CONAN_PKG::nlohmann_json CONAN_PKG::cli11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
endif()
