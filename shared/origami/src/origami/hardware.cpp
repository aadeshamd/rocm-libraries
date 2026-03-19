// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/hardware.hpp"
#include "origami/types.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace origami {

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       size_t NUM_XCD,
                       double mem1_perf_ratio,
                       double mem2_perf_ratio,
                       double mem3_perf_ratio,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       size_t parallel_mi_cu,
                       std::tuple<double, double, double> mem_bw_per_wg_coefficients)
    : arch(arch)
    , N_CU(N_CU)
    , lds_capacity(lds_capacity)
    , mem1_perf_ratio(mem1_perf_ratio)
    , mem2_perf_ratio(mem2_perf_ratio)
    , mem3_perf_ratio(mem3_perf_ratio)
    , L2_capacity(L2_capacity)
    , CU_per_L2(N_CU / NUM_XCD)
    , compute_clock_ghz(compute_clock_ghz)
    , parallel_mi_cu(parallel_mi_cu)
    , mem_bw_per_wg_coefficients(mem_bw_per_wg_coefficients)
    , NUM_XCD(NUM_XCD) {
  init_per_level_bw();
}

hardware_t::hardware_t(architecture_t arch,
                       size_t N_CU,
                       size_t lds_capacity,
                       const architecture_constants& constants,
                       size_t L2_capacity,
                       double compute_clock_ghz,
                       double memory_clock_ghz)
   : hardware_t(
          arch,
          N_CU,
          lds_capacity,
          constants.num_xcds,
          1e9 * constants.mem1_perf_ratio / (compute_clock_ghz * 1e6),
          1e9 * constants.mem2_perf_ratio / (memory_clock_ghz * 1e6 * constants.mem_clock_ratio),
          1e9 * constants.mem3_perf_ratio / (memory_clock_ghz * 1e6),
          L2_capacity,
          compute_clock_ghz,
          constants.parallel_mi_cu,
          constants.mem_bw_per_wg_coefficients) {}

hardware_t::hardware_t(hipDeviceProp_t properties)
    : hardware_t(get_hardware_for_properties(properties)) {}

hardware_t::hardware_t(const hardware_t& other)
    : arch(other.arch)
    , N_CU(other.N_CU)
    , lds_capacity(other.lds_capacity)
    , mem1_perf_ratio(other.mem1_perf_ratio)
    , mem2_perf_ratio(other.mem2_perf_ratio)
    , mem3_perf_ratio(other.mem3_perf_ratio)
    , L2_capacity(other.L2_capacity)
    , CU_per_L2(other.CU_per_L2)
    , compute_clock_ghz(other.compute_clock_ghz)
    , parallel_mi_cu(other.parallel_mi_cu)
    , mem_bw_per_wg_coefficients(other.mem_bw_per_wg_coefficients)
    , NUM_XCD(other.NUM_XCD)
    , cache_line_bytes(other.cache_line_bytes)
    , l2_bw_read(other.l2_bw_read)
    , l2_bw_write(other.l2_bw_write)
    , mall_bw_read(other.mall_bw_read)
    , mall_bw_write(other.mall_bw_write)
    , hbm_bw_read(other.hbm_bw_read)
    , hbm_bw_write(other.hbm_bw_write) {}

hardware_t hardware_t::get_hardware_for_properties(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  if (arch_enum == architecture_t::Count) {
    throw std::runtime_error(
        std::string("Attempting to retrieve hardware constants for unsupported architecture: ") +
        std::string(arch_name));
  }
  auto constants = get_arch_constants(arch_enum);
  return hardware_t(arch_enum,
                    properties.multiProcessorCount,
                    properties.sharedMemPerBlock,
                    constants,
                    properties.l2CacheSize,
                    properties.clockRate / 1.e6,
                    properties.memoryClockRate / 1.e6);
}

hardware_t hardware_t::get_hardware_for_device(int deviceId) {
  hipDeviceProp_t prop;
  hipError_t e = hipGetDeviceProperties(&prop, deviceId);
  if (e) { throw std::runtime_error(hipGetErrorString(e)); }
  return get_hardware_for_properties(prop);
}

hardware_t hardware_t::get_hardware_for_arch(architecture_t arch,
                                             size_t N_CU,
                                             size_t lds_capacity,
                                             size_t L2_capacity,
                                             int compute_clock_khz) {
  if (arch == architecture_t::Count) {
    throw std::runtime_error("Attempting to create hardware for unsupported architecture");
  }

  auto constants = get_arch_constants(arch);

  return hardware_t(arch,
                    N_CU,
                    lds_capacity,
                    constants,
                    L2_capacity,
                    compute_clock_khz / 1.e6,
                    compute_clock_khz / 1.e6 / constants.mem_clock_ratio);
}

bool hardware_t::is_hardware_supported(hipDeviceProp_t properties) {
  auto arch_name = get_before_first_colon(properties.gcnArchName);
  auto arch_enum = arch_name_to_enum(arch_name);
  return arch_enum != architecture_t::Count;
}

void hardware_t::print() const {
  std::cout << "================== Hardware Configuration ==================\n";
  std::cout << "Number of CUs (N_CU)      : " << N_CU << "\n";
  std::cout << "LDS capacity              : " << lds_capacity << " bytes\n";
  std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
  std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
  std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
  std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
  std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
  std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
  std::cout << "Parallel MI/CU            : " << parallel_mi_cu << "\n";
  std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
  std::cout << "mem_bw_per_wg_coefficients: " << std::get<0>(mem_bw_per_wg_coefficients) << ", "
            << std::get<1>(mem_bw_per_wg_coefficients) << ", "
            << std::get<2>(mem_bw_per_wg_coefficients) << "\n\n";

  std::cout << "------------------ Instruction Map -------------------------\n";
  // Loop over the instruction_map and print each entry
  for (const auto& kv : INSTRUCTION_MAP.at(arch)) {
    const auto& key  = kv.first;
    const auto& L_MI = kv.second;

    std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N << ", MI_K=" << key.MI_K
              << ", mi_input_type=" << datatype_to_string(key.mi_input_type) << " bytes\n"
              << "  -> Latency (L_MI): " << L_MI << "\n";
  }
  std::cout << "===========================================================\n";
}

size_t hardware_t::get_mi_latency(size_t MI_M,
                                  size_t MI_N,
                                  size_t MI_K,
                                  data_type_t mi_input_type) const {
  const auto& instruction_map = INSTRUCTION_MAP.at(arch);
  auto key                    = matrix_instruction(MI_M, MI_N, MI_K, mi_input_type);

  auto it = instruction_map.find(key);
  if (it != instruction_map.end()) {
    return it->second / parallel_mi_cu;
  } else {
    if (origami::runtime_options::get().debug_enabled)
      std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                << ", MI_K=" << MI_K << ", mi_input_type=" << datatype_to_string(mi_input_type)
                << ". Returning latency value of 32 (really slow).\n";
    return 32 / parallel_mi_cu;  // Default latency if instruction is not found
  }
}

bool hardware_t::has_MALL() const {
  switch (arch) {
    case architecture_t::gfx90a:
    case architecture_t::gfx942:
    case architecture_t::gfx950:
    case architecture_t::gfx1201:
    case architecture_t::gfx1100:
    case architecture_t::gfx1151: return true;
    case architecture_t::gfx1150:
    case architecture_t::gfx1152:
    case architecture_t::gfx1153: return false;
    case architecture_t::Count:
      // Count is not a valid architecture, this is to silence compiler warning
      return false;
  }
}

std::string hardware_t::get_before_first_colon(const std::string& input) {
  size_t pos = input.find(':');
  if (pos != std::string::npos) { return input.substr(0, pos); }
  return input;  // Return the whole string if ':' is not found
}

std::vector<dim3_t> hardware_t::get_valid_matrix_instructions(data_type_t mi_input_type) const {
  std::vector<dim3_t> result;

  const auto& instruction_map = INSTRUCTION_MAP.at(arch);

  for (const auto& kv : instruction_map) {
    const matrix_instruction& mi = kv.first;
    if (mi.mi_input_type == mi_input_type) { result.push_back(dim3_t{mi.MI_M, mi.MI_N, mi.MI_K}); }
  }

  return result;
}

dim3_t hardware_t::get_recommended_matrix_instruction(data_type_t mi_input_type) const {
  const auto& instruction_map = INSTRUCTION_MAP.at(arch);

  dim3_t best_dim        = {0, 0, 0};
  double best_throughput = 0.0;

  for (const auto& kv : instruction_map) {
    const matrix_instruction& mi = kv.first;
    if (mi.mi_input_type == mi_input_type) {
      size_t latency = kv.second / parallel_mi_cu;
      if (latency == 0) latency = std::numeric_limits<size_t>::max();  // Avoid division by zero

      // Calculate throughput as M*N*K/latency
      double throughput =
          static_cast<double>(mi.MI_M * mi.MI_N * mi.MI_K) / static_cast<double>(latency);

      // Update if throughput is better, or if equal, prefer instruction where M=16 (tiebreaker)
      bool is_better = throughput > best_throughput;
      bool is_tie_with_m16 =
          (throughput == best_throughput) && (mi.MI_M == 16) && (best_dim.m != 16);

      if (is_better || is_tie_with_m16) {
        best_throughput = throughput;
        best_dim        = dim3_t{mi.MI_M, mi.MI_N, mi.MI_K};
      }
    }
  }

  return best_dim;
}

void hardware_t::init_per_level_bw() {
  if (arch == architecture_t::gfx950) {
    cache_line_bytes = 64;

    // ABSOLUTE per-vector-width BW coefficients: eval_bw(coef, CUs) -> B/compute-cycle.
    // No perf_ratio needed. Coefficients derived from measured peak BW:
    //   L2 read peak:  15.5 TB/s @ 2.2 GHz = 7045 B/cycle (float4, 256 WGs)
    //   MALL read peak:  7.2 TB/s @ 2.2 GHz = 3273 B/cycle
    //   HBM read peak:   6.2 TB/s @ 2.2 GHz = 2818 B/cycle
    //   HBM write peak:  6.7 TB/s @ 2.2 GHz = 3045 B/cycle
    // Each fractional coef (a,b,c) is scaled by the peak B/cycle for its level.
    // [Short(2B), Float(4B), Float2(8B), Float4(16B)]

    constexpr double L2_PEAK   = 7045.0;   // B/cycle from l2_bandwidth test
    constexpr double MALL_PEAK = 3273.0;   // B/cycle from mall_bandwidth test
    constexpr double HBM_R_PEAK = 2818.0;  // B/cycle from stream read
    constexpr double HBM_W_PEAK = 3045.0;  // B/cycle from stream write

    auto scale = [](bw_coef_t c, double s) -> bw_coef_t {
      return std::make_tuple(std::get<0>(c) * s, std::get<1>(c) * s, std::get<2>(c) * s);
    };

    // HBM read: quadratic CU scaling (includes occupancy rolloff)
    hbm_bw_read = {{
      scale({-5.376e-06, 2.655e-03, 0.0}, HBM_R_PEAK),  // short @256≈930
      scale({-3.059e-06, 2.458e-03, 0.0}, HBM_R_PEAK),  // float @256≈1212
      scale({-7.215e-06, 4.954e-03, 0.0}, HBM_R_PEAK),  // float2 @256≈2254
      scale({-2.228e-05, 9.609e-03, 0.0}, HBM_R_PEAK)   // float4 @256≈2818
    }};

    // HBM write: higher peak than read, different per-VW profile
    hbm_bw_write = {{
      scale({-1.400e-05, 5.077e-03, 0.0}, HBM_W_PEAK),
      scale({-2.578e-05, 8.895e-03, 0.0}, HBM_W_PEAK),
      scale({-2.727e-05, 1.060e-02, 0.0}, HBM_W_PEAK),
      scale({-4.655e-05, 1.582e-02, 0.0}, HBM_W_PEAK)
    }};

    // L2 read: linear CU scaling (no quadratic term)
    l2_bw_read = {{
      scale({0.0, 6.783e-04, 0.0}, L2_PEAK),   // short @256≈1226
      scale({0.0, 1.311e-03, 0.0}, L2_PEAK),   // float @256≈2367
      scale({0.0, 2.198e-03, 0.0}, L2_PEAK),   // float2 @256≈3965
      scale({0.0, 3.906e-03, 0.0}, L2_PEAK)    // float4 @256≈7045
    }};
    l2_bw_write = {{
      scale({0.0, 2.202e-03, 0.0}, L2_PEAK),
      scale({0.0, 2.455e-03, 0.0}, L2_PEAK),
      scale({0.0, 2.604e-03, 0.0}, L2_PEAK),
      scale({0.0, 3.906e-03, 0.0}, L2_PEAK)
    }};

    // MALL read: linear CU scaling
    mall_bw_read = {{
      scale({0.0, 5.355e-04, 0.0}, MALL_PEAK),  // short @256≈449
      scale({0.0, 1.078e-03, 0.0}, MALL_PEAK),  // float @256≈903
      scale({0.0, 2.091e-03, 0.0}, MALL_PEAK),  // float2 @256≈1752
      scale({0.0, 3.906e-03, 0.0}, MALL_PEAK)   // float4 @256≈3273
    }};
    mall_bw_write = {{
      scale({0.0, 3.216e-03, 0.0}, MALL_PEAK),
      scale({0.0, 3.709e-03, 0.0}, MALL_PEAK),
      scale({0.0, 3.641e-03, 0.0}, MALL_PEAK),
      scale({0.0, 3.906e-03, 0.0}, MALL_PEAK)
    }};
  } else {
    cache_line_bytes = 128;
    // For other architectures: all vector widths use the same fractional curve.
    auto fill = [&]() -> bw_coef_array_t {
      auto c = mem_bw_per_wg_coefficients;
      return {{c, c, c, c}};
    };
    l2_bw_read    = fill();
    l2_bw_write   = fill();
    mall_bw_read  = fill();
    mall_bw_write = fill();
    hbm_bw_read   = fill();
    hbm_bw_write  = fill();
  }
}

}  // namespace origami
