#pragma once

/**
 * @file layout.h
 * @brief Pure host-side decomposition arithmetic for @ref DenseDistributed.
 *
 * Everything that decides *which cells a rank owns and how thick its halo is* lives here as plain
 * functions of (global extent, @ref ProcessGrid, boundary mode, stencil). @c DenseDistributed
 * builds its per-GPU @c Span from these, and the unit tests call the very same functions to check
 * the slab / pencil / brick geometry without a GPU or MPI.
 *
 * Splitting rule: along an axis with @c P parts, the first @c n % P coordinates receive
 * @c n / P + 1 layers and the rest @c n / P.
 *
 * Halo rule (storage thickness per side):
 * - An axis carries a halo iff it is split (@c P > 1) or periodic; its thickness is the stencil
 *   radius along that axis (@c Stencil::radius3).
 * - Slab layouts (@c Px == Py == 1) keep carbon's historical Z halo: the isotropic
 *   @c Stencil::radius, always present on Z, with X/Y halos only when those axes are periodic.
 */

#include <carbon/boundary_mode_distributed.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace carbon::detail::dense_distributed {

/**
 * Opt-in storage padding for @ref DenseDistributed spans.
 *
 * With @c pad_rows disabled (the default) the layout is exactly the historical one:
 * @c y_stride == haloed.x, no lead pad, and @c component_stride == haloed.z * xy_pitch.
 *
 * With @c pad_rows enabled, each y row is rounded up to @p row_alignment elements, each component
 * plane starts with a lead pad so buffer @c x == halo.x is @p row_alignment aligned, and the
 * component stride is rounded up to @p component_alignment elements. Both alignments are in
 * elements, not bytes; the defaults (32 and 64) are 128 B and 256 B for a 4-byte element such as
 * the fp32 LBM populations. @p component_alignment must be a multiple of @p row_alignment.
 */
struct LayoutOptions {
  bool pad_rows{false};
  int64_t row_alignment{32};
  int64_t component_alignment{64};
};

/** Round @p value up to the next multiple of @p alignment (which must be positive). */
[[nodiscard]] inline auto
round_up(int64_t value, int64_t alignment) -> int64_t {
  if (alignment <= 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: storage alignment must be positive (alignment={})", alignment);
  }
  if (value < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: cannot round a negative extent (value={})", value);
  }
  const int64_t remainder = value % alignment;
  if (remainder == 0) {
    return value;
  }
  const int64_t delta = alignment - remainder;
  if (value > std::numeric_limits<int64_t>::max() - delta) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: padded storage extent overflows int64 (value={}, alignment={})", value,
        alignment);
  }
  return value + delta;
}

/** Per-component storage strides derived from an extent and @ref LayoutOptions. */
struct StoragePadding {
  int64_t y_stride{};
  int64_t lead_pad{};
  int64_t xy_pitch{};
  int64_t component_stride{};
};

/**
 * Compute the padded storage layout for @p haloed / @p halo. The returned offsets are all in
 * elements. The default @ref LayoutOptions reproduces the historical unpadded layout exactly.
 */
[[nodiscard]] inline auto
storage_padding(const int64_3d& haloed,
                const int64_3d& halo,
                const LayoutOptions& layout = {}) -> StoragePadding {
  StoragePadding p;
  p.y_stride = haloed.x;
  p.lead_pad = 0;
  p.xy_pitch = haloed.x * haloed.y;
  p.component_stride = haloed.z * p.xy_pitch;
  if (!layout.pad_rows) {
    return p;
  }
  if (layout.row_alignment <= 0 || layout.component_alignment <= 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: storage alignments must be positive (row={}, component={})",
        layout.row_alignment, layout.component_alignment);
  }
  if (layout.component_alignment % layout.row_alignment != 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: component_alignment ({}) must be a multiple of row_alignment ({})",
        layout.component_alignment, layout.row_alignment);
  }
  p.y_stride = round_up(haloed.x, layout.row_alignment);
  p.lead_pad = (layout.row_alignment - (halo.x % layout.row_alignment)) % layout.row_alignment;
  p.xy_pitch = p.y_stride * haloed.y;
  p.component_stride = round_up(p.lead_pad + haloed.z * p.xy_pitch, layout.component_alignment);
  return p;
}

/** Split @p n layers over @p num_parts coordinates as evenly as possible (earlier ones get +1). */
[[nodiscard]] inline auto
split_extent(int64_t n,
             size_t num_parts) -> std::vector<size_t> {
  if (n < 0) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDistributed: extent must be non-negative (n={})", n);
  }
  if (static_cast<uint64_t>(n) > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    carbon::log::throw_with_log<std::overflow_error>(
        "DenseDistributed: extent too large for this platform (n={})", n);
  }
  const size_t n_u = static_cast<size_t>(n);
  if (num_parts == 0U) {
    return {};
  }
  std::vector<size_t> counts(num_parts);
  const size_t base = n_u / num_parts;
  const size_t rem = n_u % num_parts;
  for (size_t i = 0; i < num_parts; ++i) {
    counts[i] = base + (i < rem ? 1U : 0U);
  }
  return counts;
}

/** True for the historical Z-slab layout: @c Px == Py == 1 and neither X nor Y is periodic. */
[[nodiscard]] inline auto
is_legacy_slab(const ProcessGrid& pg,
               BoundaryModeDistributed boundary) noexcept -> bool {
  return pg.is_slab() && !is_periodic(boundary, Axis::X) && !is_periodic(boundary, Axis::Y);
}

/** Per-side halo thickness (uniform across ranks) for @p pg / @p boundary / @p stencil. */
[[nodiscard]] inline auto
halo_extent(const ProcessGrid& pg,
            BoundaryModeDistributed boundary,
            const Stencil& stencil) -> int64_3d {
  const int32_3d dims = pg.dims();
  const int32_3d radius3 = stencil.radius3();
  const auto axis_halo = [&](int32_t radius, int32_t parts, Axis axis) -> int64_t {
    return (parts > 1 || is_periodic(boundary, axis)) ? static_cast<int64_t>(radius) : 0;
  };
  if (is_legacy_slab(pg, boundary)) {
    return int64_3d{0, 0, static_cast<int64_t>(stencil.radius())};
  }
  return int64_3d{
      axis_halo(radius3.x, dims.x, Axis::X),
      axis_halo(radius3.y, dims.y, Axis::Y),
      axis_halo(radius3.z, dims.z, Axis::Z),
  };
}

/**
 * Validate that every global chunk is thick enough for its halo. Throws on the first offending
 * chunk in the same deterministic order on every rank, so a bad decomposition cannot throw on
 * only some ranks and leave the others waiting in the first collective.
 */
inline void
validate_global_halo_fit(const int64_3d& global,
                         const ProcessGrid& pg,
                         const int64_3d& halo,
                         bool legacy_slab,
                         const Stencil& stencil) {
  const int32_3d dims = pg.dims();
  for (const size_t nx : split_extent(global.x, static_cast<size_t>(dims.x))) {
    for (const size_t ny : split_extent(global.y, static_cast<size_t>(dims.y))) {
      for (const size_t nz : split_extent(global.z, static_cast<size_t>(dims.z))) {
        if (legacy_slab && halo.z > 0 && nz < static_cast<size_t>(2 * halo.z)) {
          carbon::log::throw_with_log<std::invalid_argument>(
              "DenseDistributed: global slab z-extent {} is smaller than 2 * halo_z = {} (stencil "
              "radius {}); the Internal / Boundary DataView split would overlap. Either grow "
              "the slab or shrink the stencil",
              nz, 2 * halo.z, stencil.radius());
        }
        if (!legacy_slab &&
            ((halo.x > 0 && nx < static_cast<size_t>(halo.x)) ||
             (halo.y > 0 && ny < static_cast<size_t>(halo.y)) ||
             (halo.z > 0 && nz < static_cast<size_t>(halo.z)))) {
          carbon::log::throw_with_log<std::invalid_argument>(
              "DenseDistributed: global owned extent ({}, {}, {}) is smaller than halo ({}, {}, "
              "{}) on a multi-axis decomposition; the halo would overlap the owned region",
              nx, ny, nz, halo.x, halo.y, halo.z);
        }
      }
    }
  }
}

/** Geometry of the chunk owned by one process-grid coordinate. */
struct RankChunk {
  /** Owned (non-halo) extent. */
  int64_3d owned{};
  /** Global coordinate of the first *haloed* cell (i.e. owned origin minus the halo). */
  int64_3d offset{};
  /** Halo thickness per side. */
  int64_3d halo{};
  /** Storage extent: owned + 2 * halo on each axis. */
  int64_3d haloed{};
  /** Elements between adjacent y rows; @c haloed.x when row padding is disabled. */
  int64_t y_stride{};
  /** Elements per XY plane, including row padding when enabled. */
  int64_t xy_pitch{};
  /** Elements per component plane, including lead pad and tail padding. */
  int64_t component_stride{};
  /** Elements before buffer @c x == 0 in each component plane. */
  int64_t lead_pad{};
};

/** Chunk owned by process coordinate @p coords of @p pg over @p global, with halo @p halo. */
[[nodiscard]] inline auto
rank_chunk(const int64_3d& global,
           const ProcessGrid& pg,
           int32_3d coords,
           const int64_3d& halo,
           const LayoutOptions& layout = {}) -> RankChunk {
  const int32_3d dims = pg.dims();
  const auto locate = [](int64_t n, int32_t parts, int32_t coord) -> std::pair<int64_t, int64_t> {
    const std::vector<size_t> counts = split_extent(n, static_cast<size_t>(parts));
    int64_t prefix = 0;
    for (int32_t i = 0; i < coord; ++i) {
      prefix += static_cast<int64_t>(counts[static_cast<size_t>(i)]);
    }
    return {static_cast<int64_t>(counts[static_cast<size_t>(coord)]), prefix};
  };
  const auto [nx, ox] = locate(global.x, dims.x, coords.x);
  const auto [ny, oy] = locate(global.y, dims.y, coords.y);
  const auto [nz, oz] = locate(global.z, dims.z, coords.z);

  RankChunk c;
  c.owned = {nx, ny, nz};
  c.halo = halo;
  c.offset = {ox - halo.x, oy - halo.y, oz - halo.z};
  c.haloed = {nx + 2 * halo.x, ny + 2 * halo.y, nz + 2 * halo.z};
  const StoragePadding padding = storage_padding(c.haloed, c.halo, layout);
  c.y_stride = padding.y_stride;
  c.xy_pitch = padding.xy_pitch;
  c.component_stride = padding.component_stride;
  c.lead_pad = padding.lead_pad;
  return c;
}

}  // namespace carbon::detail::dense_distributed
