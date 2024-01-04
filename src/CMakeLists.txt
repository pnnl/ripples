if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  add_subdirectory(gpu)
endif()

add_library(infmax STATIC imm.cc)
target_link_libraries(infmax
                      CLI11::CLI11 libtrng::libtrng fmt::fmt
                      spdlog::spdlog nlohmann_json::nlohmann_json)
if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  set_source_files_properties(imm.cc PROPERTIES LANGUAGE CUDA)
  target_link_libraries(infmax gpu_ripples)
endif()
