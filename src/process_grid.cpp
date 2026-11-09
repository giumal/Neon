#include <carbon/process_grid.h>

#include <carbon/logging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace carbon {

namespace {

constexpr std::array<int, 3> kAxisOrderZYX{2, 1, 0};

[[nodiscard]] auto
prime_factors_descending(int32_t n) -> std::vector<int32_t> {
  std::vector<int32_t> factors;
  for (int32_t p = 2; static_cast<int64_t>(p) * p <= n; ++p) {
    while (n % p == 0) {
      factors.push_back(p);
      n /= p;
    }
  }
  if (n > 1) {
    factors.push_back(n);
  }
  std::sort(factors.begin(), factors.end(), [](int32_t a, int32_t b) {
    return a > b;
  });
  return factors;
}

[[nodiscard]] auto
factorized_dims(int32_t world_size,
                DecompositionMode mode) -> int32_3d {
  std::array<int32_t, 3> dims{1, 1, 1};
  std::array<bool, 3> free_axis{false, false, false};

  switch (mode) {
    case DecompositionMode::Slab:
      free_axis[2] = true;
      break;
    case DecompositionMode::Pencil:
      free_axis[1] = true;
      free_axis[2] = true;
      break;
    case DecompositionMode::Brick:
      free_axis[0] = true;
      free_axis[1] = true;
      free_axis[2] = true;
      break;
    case DecompositionMode::Custom:
      log::throw_with_log<std::invalid_argument>(
          "carbon::ProcessGrid::make: Custom mode requires ProcessGrid::custom()");
      return {1, 1, 1};
  }

  for (const int32_t factor : prime_factors_descending(world_size)) {
    int best_axis = -1;
    for (const int axis : kAxisOrderZYX) {
      if (!free_axis[axis]) {
        continue;
      }
      if (best_axis < 0 || dims[axis] < dims[best_axis]) {
        best_axis = axis;
      }
    }
    if (best_axis < 0) {
      log::throw_with_log<std::invalid_argument>(
          "carbon::ProcessGrid::make: decomposition mode {} has no free axis",
          static_cast<int>(mode));
      return {1, 1, 1};
    }
    dims[best_axis] *= factor;
  }

  return {dims[0], dims[1], dims[2]};
}

[[nodiscard]] auto
extent_aware_dims(int32_t world_size,
                  DecompositionMode mode,
                  int64_3d global_extent) -> int32_3d {
  if (global_extent.x <= 0 || global_extent.y <= 0 || global_extent.z <= 0) {
    return factorized_dims(world_size, mode);
  }

  const auto ceil_div = [](int64_t n, int32_t parts) -> int64_t {
    return (n + static_cast<int64_t>(parts) - 1) / static_cast<int64_t>(parts);
  };

  bool found = false;
  int32_3d best{1, 1, 1};
  long double best_surface = std::numeric_limits<long double>::infinity();
  int best_split_penalty = 0;
  int32_t best_py = 0;
  int32_t best_pz = 0;

  for (int32_t px = 1; px <= world_size; ++px) {
    if (world_size % px != 0) {
      continue;
    }
    if (mode != DecompositionMode::Brick && px != 1) {
      continue;
    }
    const int32_t remaining = world_size / px;
    for (int32_t py = 1; py <= remaining; ++py) {
      if (remaining % py != 0) {
        continue;
      }
      const int32_t pz = remaining / py;
      if (mode == DecompositionMode::Slab && py != 1) {
        continue;
      }

      const int64_t lx = ceil_div(global_extent.x, px);
      const int64_t ly = ceil_div(global_extent.y, py);
      const int64_t lz = ceil_div(global_extent.z, pz);
      if (lx <= 0 || ly <= 0 || lz <= 0) {
        continue;
      }

      long double surface = 0.0L;
      if (px > 1) {
        surface += 2.0L * static_cast<long double>(ly) * static_cast<long double>(lz);
      }
      if (py > 1) {
        surface += 2.0L * static_cast<long double>(lx) * static_cast<long double>(lz);
      }
      if (pz > 1) {
        surface += 2.0L * static_cast<long double>(lx) * static_cast<long double>(ly);
      }

      const int split_axes = (px > 1 ? 1 : 0) + (py > 1 ? 1 : 0) + (pz > 1 ? 1 : 0);
      // Prefer pencil-like two-axis splits on ties: they avoid the uncoalesced X faces
      // without adding a third exchange axis.
      const int split_penalty = (split_axes == 2) ? 0 : (split_axes == 1 ? 1 : 2);

      if (!found || surface < best_surface ||
          (surface == best_surface &&
           (split_penalty < best_split_penalty ||
            (split_penalty == best_split_penalty &&
             (pz > best_pz || (pz == best_pz && py > best_py)))))) {
        found = true;
        best = {px, py, pz};
        best_surface = surface;
        best_split_penalty = split_penalty;
        best_py = py;
        best_pz = pz;
      }
    }
  }

  return found ? best : factorized_dims(world_size, mode);
}

}  // namespace

auto
ProcessGrid::make(int32_t world_size,
                  DecompositionMode mode) -> ProcessGrid {
  if (world_size <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: world_size must be > 0 (got {})", world_size);
    return ProcessGrid{};
  }
  if (mode == DecompositionMode::Custom) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: Custom mode requires ProcessGrid::custom()");
    return ProcessGrid{};
  }
  return ProcessGrid{factorized_dims(world_size, mode), mode};
}

auto
ProcessGrid::make(int32_t world_size,
                  DecompositionMode mode,
                  int64_3d global_extent) -> ProcessGrid {
  if (world_size <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: world_size must be > 0 (got {})", world_size);
    return ProcessGrid{};
  }
  if (mode == DecompositionMode::Custom) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::make: Custom mode requires ProcessGrid::custom()");
    return ProcessGrid{};
  }
  return ProcessGrid{extent_aware_dims(world_size, mode, global_extent), mode};
}

auto
ProcessGrid::custom(int32_t world_size,
                    int32_t px,
                    int32_t py,
                    int32_t pz) -> ProcessGrid {
  if (world_size <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: world_size must be > 0 (got {})", world_size);
    return ProcessGrid{};
  }
  if (px <= 0 || py <= 0 || pz <= 0) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: all dimensions must be > 0 (got px={}, py={}, pz={})",
        px,
        py,
        pz);
    return ProcessGrid{};
  }

  const int64_t xy = static_cast<int64_t>(px) * py;
  if (xy > world_size) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: px*py ({}) exceeds world_size ({})", xy, world_size);
    return ProcessGrid{};
  }
  const int64_t xyz = xy * pz;
  if (xyz != world_size) {
    log::throw_with_log<std::invalid_argument>(
        "carbon::ProcessGrid::custom: px*py*pz ({}) != world_size ({})", xyz, world_size);
    return ProcessGrid{};
  }

  return ProcessGrid{int32_3d{px, py, pz}, DecompositionMode::Custom};
}

auto
ProcessGrid::coords_of(int32_t rank) const -> int32_3d {
  if (rank < 0 || rank >= world_size()) {
    log::throw_with_log<std::out_of_range>(
        "carbon::ProcessGrid::coords_of: rank {} out of range [0, {})", rank, world_size());
    return {-1, -1, -1};
  }

  const int32_t pz = rank % dims_.z;
  const int32_t py = (rank / dims_.z) % dims_.y;
  const int32_t px = rank / (dims_.y * dims_.z);
  return {px, py, pz};
}

auto
ProcessGrid::rank_of(int32_3d coords) const -> int32_t {
  if (coords.x < 0 || coords.x >= dims_.x || coords.y < 0 || coords.y >= dims_.y ||
      coords.z < 0 || coords.z >= dims_.z) {
    log::throw_with_log<std::out_of_range>(
        "carbon::ProcessGrid::rank_of: coords ({}, {}, {}) outside grid ({}, {}, {})",
        coords.x,
        coords.y,
        coords.z,
        dims_.x,
        dims_.y,
        dims_.z);
    return -1;
  }

  return static_cast<int32_t>((static_cast<int64_t>(coords.x) * dims_.y + coords.y) * dims_.z +
                              coords.z);
}

}  // namespace carbon