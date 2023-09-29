add_library(gpu_ripples
            $<IF:${RIPPLES_ENABLE_CUDA}, LANGUAGE CUDA,>
            gpu_graph.cc lt_kernel.cc find_most_influential.cc generate_rrr_sets.cc)
target_compile_definitions(gpu_ripples
                           $<IF:${RIPPLES_ENABLE_CUDA}, RIPPLES_ENABLE_CUDA=1,>
                           $<IF:${RIPPLES_ENABLE_HIP}, RIPPLES_ENABLE_HIP=1,>)
target_link_libraries(gpu_ripples
                      CONAN_PKG::nlohmann_json CONAN_PKG::libtrng CONAN_PKG::fmt
                      CONAN_PKG::spdlog CUDA CUDART)
