if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  add_subdirectory(gpu)
endif()

add_library(infmax STATIC imm.cc)
target_link_libraries(infmax
                      $<IF:$<BOOL:${RIPPLES_ENABLE_TCMALLOC}>, gperftools::tcmalloc_minimal,>
                      CLI11::CLI11 libtrng::libtrng fmt::fmt
                      spdlog::spdlog nlohmann_json::nlohmann_json)
target_compile_definitions(infmax PRIVATE
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>)
if(RIPPLES_ENABLE_CUDA)
set_source_files_properties(imm.cc PROPERTIES LANGUAGE CUDA)
endif()
if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  target_link_libraries(infmax gpu_ripples)
endif()
