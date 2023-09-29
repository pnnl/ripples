if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  add_subdirectory(gpu)
endif()

add_library(infmax STATIC imm.cc)
target_link_libraries(infmax
                      CONAN_PKG::cli11 CONAN_PKG::libtrng CONAN_PKG::fmt
                      CONAN_PKG::spdlog CONAN_PKG::nlohmann_json)
if(RIPPLES_ENABLE_CUDA OR RIPPLES_ENABLE_HIP)
  target_link_libraries(infmax gpu_ripples)
endif()
