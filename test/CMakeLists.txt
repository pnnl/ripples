add_library(test_main test_main.cc)
target_link_libraries(test_main Catch2::Catch2)

add_executable(pivoting-tests pivoting.cc)
target_link_libraries(pivoting-tests
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME Pivoting COMMAND pivoting-tests)

add_executable(community-extraction-tests community_extraction.cc)
target_link_libraries(community-extraction-tests
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME CommunityExtraction COMMAND community-extraction-tests)

add_executable(rrr-set-generation-tests rrr_set_generation.cc)
if (RIPPLES_ENABLE_CUDA)
  set_source_files_properties(rrr_set_generation.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(imm CUDA::cudart)
endif()
target_link_libraries(rrr-set-generation-tests
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
                      fmt::fmt CLI11::CLI11 OpenMP::OpenMP_CXX infmax test_main)
target_compile_definitions(rrr-set-generation-tests PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>)

add_test(NAME RRRSetGeneration COMMAND rrr-set-generation-tests)

if (RIPPLES_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
