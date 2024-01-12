#include <string>
#include <vector>

#include "omp.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "ripples/configuration.h"
#include "ripples/find_most_influential.h"
#include "ripples/graph.h"
#include "ripples/imm_configuration.h"
#include "ripples/imm_interface.h"

#include "networkit/GlobalState.hpp"
#include "networkit/generators/BarabasiAlbertGenerator.hpp"
#include "networkit/generators/LFRGenerator.hpp"
#include "networkit/generators/RmatGenerator.hpp"
#include "networkit/generators/WattsStrogatzGenerator.hpp"
#include "networkit/graph/Graph.hpp"

template <typename GeneratorTy, typename ConfigurationTy>
void benchmark(const std::string &report_dir, const std::string &modelName,
               GeneratorTy &&G, ConfigurationTy & CFG, size_t numRRRsets) {
  ankerl::nanobench::Bench bench;
  for (auto scale : {10, 11, 12, 13, 14, 16, 17, 18, 19, 20}) {
    std::string report_file =
        report_dir + "/" + modelName + "-" + std::to_string(scale) + ".json";
    std::ofstream OS(report_file);

    auto gen = G(scale);
    NetworKit::Graph G = gen.generate();

    auto edgeRange = G.edgeRange();
    std::vector<ripples::Edge<uint32_t, float>> EL;
    for (auto itr = edgeRange.begin(); itr != edgeRange.end(); ++itr) {
      const auto &e = *itr;
      EL.push_back({uint32_t(e.u), uint32_t(e.v), 0.5f});
    }

    trng::lcg64 generator;
    generator.seed(0UL);
    generator.split(2, 1);

    using dest_type = ripples::WeightedDestination<uint32_t, float>;
    using GraphFwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::BackwardDirection<uint32_t>>;
    GraphFwd Gfwd(EL.begin(), EL.end(), false);
    GraphBwd Gbwd = Gfwd.get_transpose();

    std::vector<ripples::RRRset<GraphBwd>> RRRsets(numRRRsets);
    ripples::IMMExecutionRecord record;

    size_t num_threads;
#pragma omp single
    num_threads = omp_get_max_threads();

    ripples::ICStreamingGenerator se(Gbwd, generator, num_threads, 0, 0, 0,
                                     64,
                                     std::unordered_map<size_t, size_t>());

    ripples::GenerateRRRSets(Gbwd, se, RRRsets.begin(), RRRsets.end(), record,
                             ripples::independent_cascade_tag{},
                             ripples::omp_parallel_tag{});

    bench.complexityN(scale)
        .title("Seed Selection no Atomics")
        .performanceCounters(true)
        .output(nullptr)
        .run(modelName,
             [&]() {
               auto r = ripples::FindMostInfluentialSet(
                   Gbwd, CFG, RRRsets.begin(), RRRsets.end(), record,
                   CFG.seed_select_max_gpu_workers != 0,
                   ripples::omp_parallel_tag{});
               ankerl::nanobench::doNotOptimizeAway(r);
             })
        .render(ankerl::nanobench::templates::json(), OS);
  }
}

int main(int argc, char **argv) {
  NetworKit::GlobalState::setLogLevel(Aux::Log::LogLevel::QUIET);
  spdlog::set_level(spdlog::level::off);

  CLI::App app{"Seed Selection Benchmark"};
  std::string report_dir;
  app.add_option("-o", report_dir, "The output directory for report files.")
      ->required()
      ->check(CLI::ExistingDirectory);

  size_t numRRRSets{10000};
  app.add_option("-n", numRRRSets, "The number of RRR sets used during the experiment.")
    ->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  ripples::ToolConfiguration<ripples::IMMConfiguration> CFG;
  CFG.k = 10;
#pragma omp parallel
  {
#pragma omp single
    CFG.seed_select_max_workers = omp_get_num_threads();
  }
  CFG.seed_select_max_gpu_workers = 0;
  benchmark(report_dir, "RMAT", [](int scale) {
    return NetworKit::RmatGenerator(scale, 16, .57, .19, .19, .05);
  }, CFG, numRRRSets);
  benchmark(report_dir, "BarabasiAlbert", [](int scale) {
    return NetworKit::BarabasiAlbertGenerator(8, 1 << scale);
  }, CFG, numRRRSets);
  benchmark(report_dir, "LFR", [](int scale) {
    auto G = NetworKit::LFRGenerator(1 << scale);
    G.generatePowerlawDegreeSequence(5, 6, -2);
    G.generatePowerlawCommunitySizeSequence(5, 6, -1);
    G.setMu(.5);
    return G;
  }, CFG, numRRRSets);
  benchmark(report_dir, "WattsStrogatz", [](int scale) {
    return NetworKit::WattsStrogatzGenerator(1 << scale, 8, 0.5);
  }, CFG, numRRRSets);

#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
  std::cout << ripples::GPURuntimeTrait<RUNTIME>::num_devices() << std::endl;
  CFG.streaming_workers = 1 + ripples::GPURuntimeTrait<RUNTIME>::num_devices();
  CFG.streaming_gpu_workers = ripples::GPURuntimeTrait<RUNTIME>::num_devices();
  CFG.seed_select_max_workers =
      1 + ripples::GPURuntimeTrait<RUNTIME>::num_devices();
  CFG.seed_select_max_gpu_workers = ripples::GPURuntimeTrait<RUNTIME>::num_devices();
  benchmark(report_dir, "RMAT+GPUs", [](int scale) {
    return NetworKit::RmatGenerator(scale, 16, .57, .19, .19, .05);
  }, CFG, numRRRSets);
  benchmark(report_dir, "BarabasiAlbert+GPUs", [](int scale) {
    return NetworKit::BarabasiAlbertGenerator(8, 1 << scale);
  }, CFG, numRRRSets);
  benchmark(report_dir, "LFR+GPUs", [](int scale) {
    auto G = NetworKit::LFRGenerator(1 << scale);
    G.generatePowerlawDegreeSequence(5, 6, -2);
    G.generatePowerlawCommunitySizeSequence(5, 6, -1);
    G.setMu(.5);
    return G;
  }, CFG, numRRRSets);
  benchmark(report_dir, "WattsStrogatz+GPUs", [](int scale) {
    return NetworKit::WattsStrogatzGenerator(1 << scale, 8, 0.5);
  }, CFG, numRRRSets);
#endif

  return 0;
}
