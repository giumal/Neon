// Distributed reduction throughput benchmark for DenseDistributed.
//
// One MPI rank per GPU. Times repeated Sum reductions including the cross-rank
// NCCL stage, so it isolates the Patch 4 reduction path from LBM/CG work.
//
// Prints:
//   REDUCE_DIST_GBs=... ELEMENTS_PER_S=... elapsed_seconds=... decomposition=... gpus=N

#include <carbon/carbon.h>
#include <carbon/dense_distributed.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>
#include "bench_report.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Args {
  int64_t n = 256;
  int iters = 500;
  int warmup = 50;
  int repeats = 5;
  int batch_iters = 1;
  std::string decomposition = "slab";
  std::string report_path{};
  std::string label{};
};

[[nodiscard]] auto
parse_decomposition(std::string_view s) -> std::optional<carbon::DecompositionMode> {
  if (s == "slab" || s.empty()) {
    return carbon::DecompositionMode::Slab;
  }
  if (s == "pencil") {
    return carbon::DecompositionMode::Pencil;
  }
  if (s == "brick") {
    return carbon::DecompositionMode::Brick;
  }
  return std::nullopt;
}

[[nodiscard]] auto
parse_args(int argc,
           char** argv) -> Args {
  Args a{};
  a.n = static_cast<int64_t>(
      carbon::bench::parse_int_flag("--n", argc, argv, static_cast<int>(a.n)));
  a.iters = carbon::bench::parse_int_flag("--iters", argc, argv, a.iters);
  a.warmup = carbon::bench::parse_int_flag("--warmup", argc, argv, a.warmup);
  a.repeats = carbon::bench::parse_int_flag("--repeats", argc, argv, a.repeats);
  a.batch_iters = carbon::bench::parse_int_flag("--batch-iters", argc, argv, a.batch_iters);
  const std::string decomp = carbon::bench::parse_string_flag("--decomposition", argc, argv);
  if (!decomp.empty()) {
    a.decomposition = decomp;
  }
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  a.label = carbon::bench::parse_string_flag("--label", argc, argv);
  return a;
}

template <typename Grid>
[[nodiscard]] auto
owned_cells(const Grid& grid) -> int64_t {
  int64_t total = 0;
  for (size_t g = 0; g < grid.gpu_count(); ++g) {
    total += grid.spans()[g].iter_extent(carbon::DataView::Standard).volume();
  }
  return total;
}

template <typename Grid>
auto
bench_once(Grid& grid,
           carbon::System& sys,
           carbon::ReductionResult<float>& result,
           const Args& args) -> double {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;

  auto field = grid.template newField<float, 1>(carbon::MemorySide::HostDevice, "f");
  field.fill_owned_host([](carbon::int64_3d /*g*/, int /*c*/) { return 1.0F; });

  carbon::Stream stream = sys.stream();
  carbon::Transfer<float> upload(sys, field.cpu_buffers(), field.gpu_buffers());
  upload(stream);
  stream.wait();

  carbon::Kernel reducer = carbon::Kernel::reduce<float>(
      grid,
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HOST_DEVICE(Span s, Idx idx) -> float { return in.read(s, idx, 0); };
      },
      result, carbon::reduce::Sum<float>{}, "sum", 0U);

  reducer.launch(stream);
  result.sync(stream);
  stream.wait();
  const double expected = static_cast<double>(args.n) * static_cast<double>(args.n) *
                          static_cast<double>(args.n);
  const double got = static_cast<double>(result.host_value());
  if (std::abs(got - expected) > 1.0e-6 * expected) {
    throw std::runtime_error("reduce_dist_throughput: sum mismatch");
  }

  for (int i = 0; i < args.warmup; ++i) {
    for (int b = 0; b < args.batch_iters; ++b) {
      reducer.launch(stream);
      result.sync(stream);
    }
    stream.wait();
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < args.iters; ++i) {
    for (int b = 0; b < args.batch_iters; ++b) {
      reducer.launch(stream);
      result.sync(stream);
    }
    stream.wait();
  }
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const Args args = parse_args(argc, argv);

  if (args.n < 4 || args.iters <= 0 || args.warmup < 0 || args.repeats <= 0 ||
      args.batch_iters <= 0) {
    std::cerr << "reduce_dist_throughput: invalid --n/--iters/--warmup/--repeats/--batch-iters\n";
    return EXIT_FAILURE;
  }
  const auto dec_mode = parse_decomposition(args.decomposition);
  if (!dec_mode) {
    std::cerr << "reduce_dist_throughput: --decomposition must be slab, pencil, or brick\n";
    return EXIT_FAILURE;
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world = distributed->world_size();
  const int world_rank = distributed->world_rank();
  carbon::System sys(distributed);

  const carbon::int64_3d global{args.n, args.n, args.n};
  try {
    const carbon::ProcessGrid pg = carbon::ProcessGrid::make(world, *dec_mode);
    carbon::DenseDistributed grid(sys, global, pg, carbon::Stencil{},
                                  carbon::BoundaryModeDistributed::Open);
    const int64_t cells_local = owned_cells(grid);
    if (cells_local <= 0) {
      std::cerr << "reduce_dist_throughput: empty local domain\n";
      return EXIT_FAILURE;
    }

    carbon::ReductionResult<float> result(sys);
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.repeats));
    for (int r = 0; r < args.repeats; ++r) {
      samples.push_back(bench_once(grid, sys, result, args));
    }
    const carbon::bench::TimingStats stats =
        carbon::bench::compute_timing_stats(std::move(samples));

    const double cells_global =
        static_cast<double>(args.n) * static_cast<double>(args.n) * static_cast<double>(args.n);
    const double total_reductions =
        cells_global * static_cast<double>(args.iters) * static_cast<double>(args.batch_iters);
    const double eps = total_reductions / stats.median_seconds;
    const double gbs = eps * static_cast<double>(sizeof(float)) / 1.0e9;

    if (world_rank == 0) {
      std::cout << "REDUCE_DIST_GBs=" << gbs << " ELEMENTS_PER_S=" << eps
                << " elapsed_seconds=" << stats.median_seconds
                << " decomposition=" << args.decomposition
                << " gpus=" << world << " n=" << args.n << " iters=" << args.iters
                << " batch_iters=" << args.batch_iters << " repeats=" << args.repeats
                << " cells_local=" << cells_local << " sum_ok=1\n";

      if (!args.report_path.empty()) {
        std::ostringstream cfg;
        cfg << "{\n";
        cfg << "    \"n\": " << args.n << ",\n";
        cfg << "    \"iters\": " << args.iters << ",\n";
        cfg << "    \"warmup\": " << args.warmup << ",\n";
        cfg << "    \"repeats\": " << args.repeats << ",\n";
        cfg << "    \"batch_iters\": " << args.batch_iters << ",\n";
        cfg << "    \"decomposition\": \"" << args.decomposition << "\",\n";
        cfg << "    \"label\": \"" << carbon::bench::json_escape(args.label) << "\"\n";
        cfg << "  }";
        if (!carbon::bench::write_json_report(args.report_path, "reduce_dist_throughput",
                                              cfg.str(), sys, stats, gbs, "reduce_gbs")) {
          return EXIT_FAILURE;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "reduce_dist_throughput: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
#else
  std::cout << "reduce_dist_throughput: built without MPI/NCCL, skipping\n";
#endif
  return EXIT_SUCCESS;
}
