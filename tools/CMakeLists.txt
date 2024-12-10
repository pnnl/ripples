add_executable(imm imm.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(imm.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(imm CUDA::cudart)
endif()
target_link_libraries(imm
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      infmax)
if (Metall_FOUND)
  target_compile_definitions(imm PRIVATE ENABLE_METALL=1)
  if(RIPPLES_ENABLE_METALL_CHECKPOINTING)
    target_compile_definitions(imm PRIVATE ENABLE_METALL_CHECKPOINTING=1)
  endif()
  target_link_libraries(imm Metall::Metall)
endif()

target_compile_definitions(imm PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_FLOAT_WEIGHTS}>, RIPPLES_ENABLE_FLOAT_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT16_WEIGHTS}>, RIPPLES_ENABLE_UINT16_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT8_WEIGHTS}>, RIPPLES_ENABLE_UINT8_WEIGHTS=1,>)

add_executable(opimc opimc.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(opimc.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(opimc CUDA::cudart)
endif()
target_link_libraries(opimc
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      infmax)
if (Metall_FOUND)
  target_compile_definitions(opimc PRIVATE ENABLE_METALL=1)
  if(RIPPLES_ENABLE_METALL_CHECKPOINTING)
    target_compile_definitions(opimc PRIVATE ENABLE_METALL_CHECKPOINTING=1)
  endif()
  target_link_libraries(opimc Metall::Metall)
endif()

target_compile_definitions(opimc PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_FLOAT_WEIGHTS}>, RIPPLES_ENABLE_FLOAT_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT16_WEIGHTS}>, RIPPLES_ENABLE_UINT16_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT8_WEIGHTS}>, RIPPLES_ENABLE_UINT8_WEIGHTS=1,>)


add_executable(simulator simulator.cc)
target_link_libraries(simulator
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

add_executable(dump-graph dump-graph.cc)
target_link_libraries(dump-graph
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

target_compile_definitions(dump-graph PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_FLOAT_WEIGHTS}>, RIPPLES_ENABLE_FLOAT_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT16_WEIGHTS}>, RIPPLES_ENABLE_UINT16_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT8_WEIGHTS}>, RIPPLES_ENABLE_UINT8_WEIGHTS=1,>)


add_executable(hill-climbing hill_climbing.cc)
target_link_libraries(hill-climbing
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX)

if (MPI_FOUND)
add_executable(mpi-opimc mpi-opimc.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(mpi-opimc.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(mpi-opimc CUDA::cudart)
endif()
target_link_libraries(mpi-opimc
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX infmax)

target_compile_definitions(mpi-opimc PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_FLOAT_WEIGHTS}>, RIPPLES_ENABLE_FLOAT_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT16_WEIGHTS}>, RIPPLES_ENABLE_UINT16_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT8_WEIGHTS}>, RIPPLES_ENABLE_UINT8_WEIGHTS=1,>)

add_executable(mpi-imm mpi-imm.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(mpi-imm.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(mpi-imm CUDA::cudart)
endif()
target_link_libraries(mpi-imm
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX infmax)

target_compile_definitions(mpi-imm PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_FLOAT_WEIGHTS}>, RIPPLES_ENABLE_FLOAT_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT16_WEIGHTS}>, RIPPLES_ENABLE_UINT16_WEIGHTS=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_UINT8_WEIGHTS}>, RIPPLES_ENABLE_UINT8_WEIGHTS=1,>)
if (Metall_FOUND)
  target_compile_definitions(mpi-imm PRIVATE ENABLE_METALL=1)
  if(RIPPLES_ENABLE_METALL_CHECKPOINTING)
    target_compile_definitions(mpi-imm PRIVATE ENABLE_METALL_CHECKPOINTING=1)
  endif()
  target_link_libraries(mpi-imm Metall::Metall)
endif()

add_executable(mpi-hill-climbing mpi-hill-climbing.cc)
target_link_libraries(mpi-hill-climbing
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      spdlog::spdlog fmt::fmt libtrng::libtrng
                      nlohmann_json::nlohmann_json CLI11::CLI11 OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)
endif()
