add_executable(bench-find_most_influential find_most_influential.cc)
target_link_libraries(bench-find_most_influential
                      nanobench::nanobench networkit::networkit
                      libtrng::libtrng nlohmann_json::nlohmann_json CLI11::CLI11 spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX)
