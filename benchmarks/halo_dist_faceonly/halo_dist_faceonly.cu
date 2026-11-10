// Distributed face-only halo-exchange benchmark.
//
// Times a raw HaloDistributed exchange over a DenseDistributed field with a
// six-point face stencil and StencilSemantic::LatticeDistributed. This is the
// shape that exercises the packed / fused face-only halo path; LBM D3Q19/D3Q27
// stencils contain diagonals and therefore never enter that branch.
//
// One MPI rank per GPU. Prints scrapeable lines:
//   HALO_GBs=... MS_PER_ITER=... dist_entries=... decomposition=... gpus=N

#include <carbon/carbon.h>
#include <carbon/dense_distributed.h>
#include <carbon/distributed.h>
#include <carbon/halo_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include "bench_report.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kComponents = 6;

struct Args {
  int64_t n = 256;
  int iters = 200;
  int warmup = 50;
  int repeats = 5;
  std::string decomposition = "slab";
  std::string report_path{};
  std::string label{};
};

[[nodiscard]] auto
six_point() -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
}

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
env_flag(std::string_view name,
         bool fallback) -> bool {
  const char* value = std::getenv(std::string{name}.c_str());
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "False") != 0;
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
  const std::string decomp = carbon::bench::parse_string_flag("--decomposition", argc, argv);
  if (!decomp.empty()) {
    a.decomposition = decomp;
  }
  a.report_path = carbon::bench::parse_string_flag("--report", argc, argv);
  a.label = carbon::bench::parse_string_flag("--label", argc, argv);
  return a;
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  const Args args = parse_args(argc, argv);

  if (args.n < 4 || args.iters <= 0 || args.warmup < 0 || args.warmup >= args.iters ||
      args.repeats <= 0) {
    std::cerr << "halo_dist_faceonly: invalid --n/--iters/--warmup/--repeats\n";
    return EXIT_FAILURE;
  }
  const auto dec_mode = parse_decomposition(args.decomposition);
  if (!dec_mode) {
    std::cerr << "halo_dist_faceonly: --decomposition must be slab, pencil, or brick\n";
    return EXIT_FAILURE;
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int world = distributed->world_size();
  const int world_rank = distributed->world_rank();
  if (world < 2) {
    std::cout << "halo_dist_faceonly: world_size=" << world << " < 2, skipping\n";
    return EXIT_SUCCESS;
  }
  carbon::System sys(distributed);

  const carbon::int64_3d global{args.n, args.n, args.n};
  const carbon::Stencil stencil = six_point();

  try {
    const carbon::ProcessGrid pg = carbon::ProcessGrid::make(world, *dec_mode);
    carbon::DenseDistributed grid(sys, global, pg, stencil,
                                  carbon::BoundaryModeDistributed::Open);
    auto field = grid.template newField<float, kComponents>(carbon::MemorySide::Device);
    carbon::HaloDistributed<float, kComponents> exchange(
        field, stencil, carbon::StencilSemantic::LatticeDistributed);

    if (exchange.dist_entry_count() == 0U) {
      std::cerr << "halo_dist_faceonly: no cross-rank entries (decomposition/rank mismatch?)\n";
      return EXIT_FAILURE;
    }

    carbon::Stream stream = sys.stream();
    for (int i = 0; i < args.warmup; ++i) {
      exchange(stream);
      stream.wait();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.repeats));
    for (int r = 0; r < args.repeats; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < args.iters; ++i) {
        exchange(stream);
        stream.wait();
      }
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    const carbon::bench::TimingStats stats =
        carbon::bench::compute_timing_stats(std::move(samples));
    const double seconds = stats.median_seconds;
    const double ms_per_iter = seconds * 1000.0 / static_cast<double>(args.iters);

    const auto& span = grid.spans()[0];
    const carbon::int64_3d owned = span.logic_extent();
    const carbon::int32_3d dims = pg.dims();
    int64_t face_cells = 0;
    if (dims.x > 1) {
      face_cells += 2 * owned.y * owned.z;
    }
    if (dims.y > 1) {
      face_cells += 2 * owned.x * owned.z;
    }
    if (dims.z > 1) {
      face_cells += 2 * owned.x * owned.y;
    }
    const double bytes_per_iter =
        static_cast<double>(face_cells) * kComponents * static_cast<double>(sizeof(float));
    const double gbs = bytes_per_iter * static_cast<double>(args.iters) / seconds / 1.0e9;

    const bool pack = env_flag("CARBON_HALO_PACK", true);
    const bool pack_z = env_flag("CARBON_HALO_PACK_Z", true);

    if (world_rank == 0) {
      std::cout << "HALO_GBs=" << gbs << " MS_PER_ITER=" << ms_per_iter
                << " dist_entries=" << exchange.dist_entry_count()
                << " decomposition=" << args.decomposition
                << " gpus=" << world << " n=" << args.n << " iters=" << args.iters
                << " components=" << kComponents << " pack=" << (pack ? 1 : 0)
                << " pack_z=" << (pack_z ? 1 : 0) << '\n';

      if (!args.report_path.empty()) {
        std::ostringstream cfg;
        cfg << "{\n";
        cfg << "    \"n\": " << args.n << ",\n";
        cfg << "    \"iters\": " << args.iters << ",\n";
        cfg << "    \"warmup\": " << args.warmup << ",\n";
        cfg << "    \"repeats\": " << args.repeats << ",\n";
        cfg << "    \"components\": " << kComponents << ",\n";
        cfg << "    \"decomposition\": \"" << args.decomposition << "\",\n";
        cfg << "    \"pack\": " << (pack ? "true" : "false") << ",\n";
        cfg << "    \"pack_z\": " << (pack_z ? "true" : "false") << ",\n";
        cfg << "    \"label\": \"" << carbon::bench::json_escape(args.label) << "\"\n";
        cfg << "  }";
        if (!carbon::bench::write_json_report(args.report_path, "halo_dist_faceonly",
                                              cfg.str(), sys, stats, gbs, "halo_gbs")) {
          return EXIT_FAILURE;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "halo_dist_faceonly: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
#else
  std::cout << "halo_dist_faceonly: built without MPI/NCCL, skipping\n";
#endif
  return EXIT_SUCCESS;
}
