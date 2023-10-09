add_library(gpu_ripples
            $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, LANGUAGE CUDA,>
            gpu_graph.cc lt_kernel.cc find_most_influential.cc generate_rrr_sets.cc)
target_compile_definitions(gpu_ripples PUBLIC
                           $<IF:$<BOOL:${RIPPLES_ENABLE_CUDA}>, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:$<BOOL:${RIPPLES_ENABLE_HIP}>, RIPPLES_ENABLE_HIP=1,>)
target_link_libraries(gpu_ripples
                      nlohmann_json::nlohmann_json libtrng::libtrng fmt::fmt
                      spdlog::spdlog)
if (RIPPLES_ENABLE_HIP)
  target_link_libraries(gpu_ripples rocthrust::rocthrust)
endif()
