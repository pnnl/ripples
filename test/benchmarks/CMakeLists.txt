add_executable(bench-find_most_influential find_most_influential.cc)
target_link_libraries(bench-find_most_influential
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      nanobench::nanobench networkit::networkit
                      libtrng::libtrng nlohmann_json::nlohmann_json
                      CLI11::CLI11 spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX infmax)
target_compile_definitions(bench-find_most_influential PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(find_most_influential.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(bench-find_most_influential CUDA::cudart)
endif()
