add_library(test_main test_main.cc)
target_link_libraries(test_main CONAN_PKG::catch2)

add_executable(pivoting-tests pivoting.cc)
target_link_libraries(pivoting-tests
                      CONAN_PKG::libtrng CONAN_PKG::nlohmann_json CONAN_PKG::spdlog
                      CONAN_PKG::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME Pivoting COMMAND pivoting-tests)

add_executable(community-extraction-tests community_extraction.cc)
target_link_libraries(community-extraction-tests
                      CONAN_PKG::libtrng CONAN_PKG::nlohmann_json CONAN_PKG::spdlog
                      CONAN_PKG::fmt OpenMP::OpenMP_CXX test_main)
add_test(NAME CommunityExtraction COMMAND community-extraction-tests)

# add_executable(rrr-set-generation-tests rrr_set_generation.cc)
# target_link_libraries(rrr-set-generation-tests
#                       CONAN_PKG::libtrng CONAN_PKG::nlohmann_json CONAN_PKG::spdlog
#                       CONAN_PKG::fmt CONAN_PKG::cli11 OpenMP::OpenMP_CXX test_main)
# add_test(NAME RRRSetGeneration COMMAND rrr-set-generation-tests)
