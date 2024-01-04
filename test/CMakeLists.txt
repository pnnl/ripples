add_library(test_main test_main.cc)
target_link_libraries(test_main Catch2::Catch2)

add_executable(pivoting-tests pivoting.cc)
target_link_libraries(pivoting-tests
                      libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME Pivoting COMMAND pivoting-tests)

add_executable(community-extraction-tests community_extraction.cc)
target_link_libraries(community-extraction-tests
                      libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
                      fmt::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME CommunityExtraction COMMAND community-extraction-tests)

# add_executable(rrr-set-generation-tests rrr_set_generation.cc)
# target_link_libraries(rrr-set-generation-tests
#                       libtrng::libtrng nlohmann_json::nlohmann_json spdlog::spdlog
#                       fmt::fmt CLI::CLI11 OpenMP::OpenMP_CXX test_main)
# add_test(NAME RRRSetGeneration COMMAND rrr-set-generation-tests)
