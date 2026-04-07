// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core/arch/mma/mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_selector.hpp"
#include "ck_tile/core/arch/mma/mma_traits.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"
#include "ck_tile/core/arch/mma/sparse/sparse_transforms.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include <cstdint>
#include <type_traits>

namespace ck_tile::core::arch::mma {

namespace sparse::detail {
// TODO: c++20: return MmaPipelineOptionFlags directly
constexpr inline int getPipelineFlags()
{
    return static_cast<int>(MmaPipelineOptionFlag::COMPRESS_A);
}
} // namespace sparse::detail

/**
 * @class SparseMmaPipeline
 * @brief Driver for the wave-tile sparse Mma operation. Given a backend MmaOp implementation
 * (e.g., smfmac), this class performs fragment-wise (MmaTile) decomposition to matrix-multiply
 * input WaveTiles of (A: WaveTileM x WaveTileK) x (B: WaveTileK x WaveTileN) and accumulates
 * results into output WaveTile (C: WaveTileM x WaveTileN).
 *
 * Like WaveWiseMmaPipeline, this decomposes WaveTile dimensions into fragments and iterates
 * internally over FragsM × FragsN × FragsK. The A operand is provided in uncompressed form;
 * 2:4 structured sparsity compression (SparseCompressTransform) is applied per-fragment inside
 * the execution loop.
 *
 * @tparam ADataType      Data type of input WaveTile A
 * @tparam BDataType      Data type of input WaveTile B
 * @tparam CDataType      Data type of input/output WaveTile C (accumulator)
 * @tparam WaveTileM      Mma WaveTile M dimension
 * @tparam WaveTileN      Mma WaveTile N dimension
 * @tparam WaveTileK      Mma WaveTile K dimension
 * @tparam AccumPolicy    The fragment order of the accum. registers (row or col major frag order)
 * @tparam CompilerTarget The compiler target
 * @tparam MmaOp_         Backend wrapper class that will perform the mma op (e.g., smfmac)
 * @tparam MmaTransforms  The set of transforms to be applied to input/output WaveTiles
 */
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          uint32_t WaveTileM,
          uint32_t WaveTileN,
          uint32_t WaveTileK,
          MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR,
          typename CompilerTarget =
              decltype(get_compiler_target()), // TODO: c++20 amdgcn_target_arch_id GfxTargetId =
                                               // get_compiler_target(),
          typename MmaOp_ =
              typename MmaDefaultSelector<ADataType, // TODO: c++20 MmaOpI MmaOp = typename
                                                     // MmaDefaultSelector<ADataType,
                                          BDataType,
                                          CDataType,
                                          WaveTileM,
                                          WaveTileN,
                                          WaveTileK,
                                          CompilerTarget,
                                          MmaOpFamily::SPARSE>::SelectedOp,
          typename MmaTransforms = // TODO: c++20 MmaTransformsI MmaTransforms =
          typename MmaTransformsDefaultSelector<MmaOp_, CompilerTarget>::SelectedTransforms>
// clang-format off
struct SparseMmaPipeline : public MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>
{
    using Base = MmaPipelineBase<sparse::detail::getPipelineFlags(), SparseMmaPipeline<ADataType, BDataType, CDataType, WaveTileM, WaveTileN, WaveTileK, AccumPolicy, CompilerTarget, MmaOp_, MmaTransforms>>;
    // clang-format on

    static_assert(!Base::template hasFlag<MmaPipelineOptionFlag::C_TRANSPOSE>(),
                  "Cannot transpose C in sparse intrinsics.");

    using MmaOp = MmaOp_; // Expose the selected MmaOp

    // Fragment dimensions (from the hardware MmaOp)
    constexpr static uint32_t FragM = MmaOp::kM;
    constexpr static uint32_t FragN = MmaOp::kN;
    constexpr static uint32_t FragK = MmaOp::kK;

    // Fragment counts for decomposition
    constexpr static uint32_t FragsM = WaveTileM / FragM;
    constexpr static uint32_t FragsN = WaveTileN / FragN;
    constexpr static uint32_t FragsK = WaveTileK / FragK;

    // Calculate the uncompressed external A per-fragment vector type
    struct ExternalAVecCalculator
    {
        using AVecTraits               = vector_traits<typename MmaOp::AVecType>;
        static constexpr index_t ASize = AVecTraits::vector_size * MmaOp::kCompressionRatio;
        using AVecType                 = ext_vector_t<typename AVecTraits::scalar_type, ASize>;
    };
    using ExternalAFragVecT = typename ExternalAVecCalculator::AVecType;

    // Expose internal (per-fragment) vector types
    using InternalAVecT = typename MmaOp::AVecType;
    using InternalBVecT = typename MmaOp::BVecType;
    using InternalCVecT = typename MmaOp::CVecType;

    // Buffer types for WaveTiles (caller-facing)
    using AVecType = ExternalAFragVecT[FragsM][FragsK];
    using BVecType = InternalBVecT[FragsN][FragsK];
    using CVecType = InternalCVecT[FragsM][FragsN];

    // Transforms
    using ATransform = typename MmaTransforms::ATransform;
    using BTransform = typename MmaTransforms::BTransform;
    using CTransform = typename MmaTransforms::CTransform;
    using DTransform = typename MmaTransforms::DTransform;

    // Sanity checks
    static_assert(WaveTileM >= FragM, "WaveTileM must be >= FragM");
    static_assert(WaveTileN >= FragN, "WaveTileN must be >= FragN");
    static_assert(WaveTileK >= FragK, "WaveTileK must be >= FragK");
    static_assert(WaveTileM % FragM == 0u, "WaveTileM must be a multiple of FragM");
    static_assert(WaveTileN % FragN == 0u, "WaveTileN must be a multiple of FragN");
    static_assert(WaveTileK % FragK == 0u, "WaveTileK must be a multiple of FragK");

    // Override exec() to handle per-fragment sparse compression.
    // We pre-compress all A fragments before the accumulation loop because the
    // SparseCompressTransform modifies the A vector in-place (destructive).
    template <typename VecTA, typename VecTB, typename VecTC>
    CK_TILE_DEVICE static decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC&& accum)
    {
        if constexpr(MmaOpTraits<MmaOp>::IsSupported)
        {
            auto& b_frag = b;
            auto& c_frag = accum;

            // Pre-compress all A fragments. SparseCompressTransform::exec returns a
            // tuple<CompressedVec&, int32_t> where the first element is a reference
            // into the (now-modified) a[bm][bk]. We store the idx per fragment.
            int32_t a_idx[FragsM][FragsK];
            InternalAVecT a_compressed[FragsM][FragsK];
            for(uint32_t bm = 0u; bm < FragsM; ++bm)
            {
                for(uint32_t bk = 0u; bk < FragsK; ++bk)
                {
                    auto [compressed_ref, idx] = ATransform::exec(a[bm][bk]);
                    a_compressed[bm][bk]       = compressed_ref;
                    a_idx[bm][bk]              = idx;
                }
            }

            // Accumulation loop
            if constexpr(AccumPolicy == MmaAccumPolicy::ROW_MAJOR)
            {
                for(uint32_t bm = 0u; bm < FragsM; ++bm)
                {
                    for(uint32_t bn = 0u; bn < FragsN; ++bn)
                    {
                        for(uint32_t bk = 0u; bk < FragsK; ++bk)
                        {
                            c_frag[bm][bn] = MmaOp::exec(a_compressed[bm][bk],
                                                         b_frag[bn][bk],
                                                         c_frag[bm][bn],
                                                         a_idx[bm][bk]);
                        }
                    }
                }
            }
            else if constexpr(AccumPolicy == MmaAccumPolicy::COL_MAJOR)
            {
                for(uint32_t bn = 0u; bn < FragsN; ++bn)
                {
                    for(uint32_t bm = 0u; bm < FragsM; ++bm)
                    {
                        for(uint32_t bk = 0u; bk < FragsK; ++bk)
                        {
                            c_frag[bm][bn] = MmaOp::exec(a_compressed[bm][bk],
                                                         b_frag[bn][bk],
                                                         c_frag[bm][bn],
                                                         a_idx[bm][bk]);
                        }
                    }
                }
            }
            else
            {
                static_assert(false);
            }

            // Apply D transform (pass-through for sparse) and return
            return DTransform::exec(c_frag);
        }
        else
        {
            return MmaOp::exec({}, {}, {});
        }
    }

    // execImpl is not used — exec() is overridden directly. Kept for interface compatibility.
    template <typename ATransformResult, typename BTransformResult, typename CTransformResult>
    CK_TILE_DEVICE static void
    execImpl(std::tuple<ATransformResult, BTransformResult, CTransformResult>&)
    {
        static_assert(false,
                      "SparseMmaPipeline::execImpl should not be called. Use exec() directly.");
    }
};

} // namespace ck_tile::core::arch::mma
