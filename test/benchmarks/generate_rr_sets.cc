#include <string>
#include <vector>
#include <iostream>

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

    std::cout << "Generating " << modelName << " graph with scale " << scale << std::endl;

    auto gen = G(scale);
    NetworKit::Graph G = gen.generate();

    std::cout << "Populating edge list" << std::endl;

    auto edgeRange = G.edgeRange();
    #if defined RIPPLES_ENABLE_UINT8_WEIGHTS
      using EdgeT = ripples::Edge<uint32_t, uint8_t>;
      constexpr uint8_t wt = std::numeric_limits<uint8_t>::max()/2;
    #elif defined RIPPLES_ENABLE_UINT16_WEIGHTS
      using EdgeT = ripples::Edge<uint32_t, uint16_t>;
      constexpr uint16_t wt = std::numeric_limits<uint16_t>::max()/2;
    #else
      using EdgeT = ripples::Edge<uint32_t, float>;
      constexpr float wt = 0.5;
    #endif // RIPPLES_WEIGHT_QUANT
    std::vector<EdgeT> EL;
    for (auto itr = edgeRange.begin(); itr != edgeRange.end(); ++itr) {
      const auto &e = *itr;
      EL.push_back({uint32_t(e.u), uint32_t(e.v), wt});
    }

    std::cout << EL.size() << " edges" << std::endl;

    trng::lcg64 generator;
    generator.seed(0UL);
    generator.split(2, 1);

    #if defined RIPPLES_ENABLE_UINT8_WEIGHTS
      using dest_type = ripples::WeightedDestination<uint32_t, uint8_t>;
    #elif defined RIPPLES_ENABLE_UINT16_WEIGHTS
      using dest_type = ripples::WeightedDestination<uint32_t, uint16_t>;
    #else
      using dest_type = ripples::WeightedDestination<uint32_t, float>;
    #endif // RIPPLES_WEIGHT_QUANT
    using GraphFwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::ForwardDirection<uint32_t>>;
    using GraphBwd = ripples::Graph<uint32_t, dest_type,
                                    ripples::BackwardDirection<uint32_t>>;
    std::cout << "Building forward graph" << std::endl;
    GraphFwd Gfwd(EL.begin(), EL.end(), false);

    // Traverse the graph and all of its edges, ensure ids are within numEdges and numNodes
    // for (size_t u = 0; u < Gfwd.num_nodes(); ++u) {
    //   for (auto e : Gfwd.neighbors(u)) {
    //     if (e.vertex >= Gfwd.num_nodes()) {
    //       std::cerr << "Invalid forward edge: " << u << " -> " << e.vertex << std::endl;
    //       std::cerr << "numNodes: " << Gfwd.num_nodes() << std::endl;
    //       std::cerr << "numEdges: " << Gfwd.num_edges() << std::endl;
    //       throw std::runtime_error("Invalid forward edge");
    //     }
    //   }
    // }

    std::cout << "Building backwards graph" << std::endl;
    GraphBwd Gbwd = Gfwd.get_transpose();

    // for (size_t u = 0; u < Gbwd.num_nodes(); ++u) {
    //   for (auto e : Gbwd.neighbors(u)) {
    //     if (e.vertex >= Gbwd.num_nodes()) {
    //       std::cerr << "Invalid back edge: " << u << " -> " << e.vertex << std::endl;
    //       std::cerr << "numNodes: " << Gbwd.num_nodes() << std::endl;
    //       std::cerr << "numEdges: " << Gbwd.num_edges() << std::endl;
    //       throw std::runtime_error("Invalid back edge");
    //     }
    //   }
    // }

    std::vector<ripples::RRRset<GraphBwd>> RRRsets(numRRRsets);
    ripples::IMMExecutionRecord record;

    ripples::streaming_command_line(
        CFG.worker_to_gpu, CFG.streaming_workers, CFG.streaming_cpu_teams,
        CFG.streaming_gpu_workers, CFG.gpu_mapping_string);

    ripples::ICStreamingGenerator se(
            Gbwd, generator, CFG.streaming_workers - CFG.streaming_gpu_workers,
            CFG.streaming_cpu_teams, CFG.streaming_gpu_workers,
            CFG.gpu_batch_size, CFG.cpu_batch_size,
            CFG.worker_to_gpu);
    
    std::cout << "Generating RRR sets" << std::endl;

    bench.complexityN(scale)
        .title("Sampling Array of Structs")
        .performanceCounters(true)
        .output(nullptr)
        .run(modelName,
             [&]() {
               ripples::GenerateRRRSets(
                             Gbwd, se, RRRsets.begin(), RRRsets.end(), record,
                             ripples::independent_cascade_tag{},
                             ripples::omp_parallel_tag{});
             })
        .render(ankerl::nanobench::templates::json(), OS);
    
    std::cout << "RRR Set Generation Done!" << std::endl;

    // auto r = ripples::FindMostInfluentialSet(
    //                Gbwd, CFG, RRRsets.begin(), RRRsets.end(), record,
    //                CFG.seed_select_max_gpu_workers != 0,
    //                ripples::omp_parallel_tag{});
  }
}

enum class EngineConfig { CPU, GPU, Both };

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

  EngineConfig ConfigFor{EngineConfig::CPU};
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
  std::map<std::string, EngineConfig> map{
    {"CPU", EngineConfig::CPU}, {"GPU", EngineConfig::GPU}, {"Both", EngineConfig::Both}};
  app.add_option("-c", ConfigFor,
                 "Config Streaming Engine For CPU, GPUs, or Both")
      ->capture_default_str()
      ->transform(CLI::CheckedTransformer(map, CLI::ignore_case));
#endif
  CLI11_PARSE(app, argc, argv);

  ripples::ToolConfiguration<ripples::IMMConfiguration> CFG;
  CFG.k = 10;

  size_t numThreads{0};
  numThreads = omp_get_max_threads();

  std::cout << "Using " << numThreads << " threads" << std::endl;

  switch (ConfigFor) {
#if defined(RIPPLES_ENABLE_CUDA) || defined(RIPPLES_ENABLE_HIP)
    case EngineConfig::Both:
      std::cout << "Using CPU and GPU" << std::endl;
      CFG.streaming_workers = numThreads;
      CFG.streaming_gpu_workers =
          ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      CFG.seed_select_max_workers = numThreads;
      CFG.seed_select_max_gpu_workers =
          ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      break;
    case EngineConfig::GPU:
      std::cout << "Using GPU only" << std::endl;
      CFG.streaming_workers =
          1 + ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      CFG.streaming_gpu_workers =
          ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      CFG.seed_select_max_workers =
          1 + ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      CFG.seed_select_max_gpu_workers =
          ripples::GPURuntimeTrait<RUNTIME>::num_devices();
      break;
#endif
    default:
    case EngineConfig::CPU:
      std::cout << "Using CPU only" << std::endl;
      CFG.streaming_workers = numThreads;
      CFG.streaming_gpu_workers = 0;
      CFG.streaming_cpu_teams = 0;
      CFG.seed_select_max_workers = numThreads;
      CFG.seed_select_max_gpu_workers = 0;
      CFG.cpu_batch_size = 64;
  }

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

  return 0;
}
