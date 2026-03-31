// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include <algorithm>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <vector>

#include "client_except.h"
#include "enum_to_string.h"
#include "fft_params.h"
#include "fftw_transform.h"
#include "gpubuf.h"
#include "rocfft_against_fftw.h"
#include "sys_mem.h"
#include "test_callbacks.h"
#include "test_params.h"

extern int  verbose;
extern bool fftw_compare;

static auto allocate_cpu_fft_buffer(const fft_precision        precision,
                                    const fft_array_type       type,
                                    const std::vector<size_t>& size)
{
    // FFTW does not support half-precision, so we do single instead.
    // So if we need to do a half-precision FFTW transform, allocate
    // enough buffer for single-precision instead.
    return allocate_host_buffer(
        precision == fft_precision_half ? fft_precision_single : precision, type, size);
}

template <typename Tfloat>
inline void execute_cpu_fft(const fft_params&                            cpu_fft_params,
                            typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan,
                            std::vector<hostbuf>&                        cpu_input,
                            std::vector<hostbuf>&                        cpu_output)
{
    // CPU output might not be allocated already for us, if FFTW never
    // needed an output buffer during planning
    if(cpu_output.empty())
        cpu_output = allocate_cpu_fft_buffer(
            cpu_fft_params.precision, cpu_fft_params.otype, cpu_fft_params.osize);

    // If this is either C2R or callbacks are enabled, the
    // input will be modified.  So we need to modify the copy instead.
    std::vector<hostbuf>  cpu_input_copy(cpu_input.size());
    std::vector<hostbuf>* input_ptr = &cpu_input;
    if(cpu_fft_params.run_callbacks
       || cpu_fft_params.transform_type == fft_transform_type_real_inverse)
    {
        for(size_t i = 0; i < cpu_input.size(); ++i)
        {
            cpu_input_copy[i] = cpu_input[i].copy();
        }

        input_ptr = &cpu_input_copy;
    }

    // run FFTW (which may destroy CPU input)
    apply_load_callback(cpu_fft_params, *input_ptr);
    cpu_fft_params.apply_host_load_ops(*input_ptr);
    fftw_run<Tfloat>(cpu_fft_params.transform_type, cpu_plan, *input_ptr, cpu_output);
    // clean up
    // ask FFTW to fully clean up, since it tries to cache plan details
    static_assert(
        std::is_same_v<
            typename fftw_trait<Tfloat>::fftw_plan_type,
            fftw_plan> || std::is_same_v<typename fftw_trait<Tfloat>::fftw_plan_type, fftwf_plan>);
    fftw_destroy_plan_type(cpu_plan);
    if constexpr(std::is_same_v<typename fftw_trait<Tfloat>::fftw_plan_type, fftw_plan>)
        fftw_cleanup();
    else
        fftwf_cleanup();
    cpu_plan = nullptr;
    cpu_fft_params.apply_host_store_ops(cpu_output);
    apply_store_callback(cpu_fft_params, cpu_output);
}

struct reference_fft_data_t
{
    reference_fft_data_t(const fft_params& test_params)
    {
        // If current cached results are valid reference results for the given
        // parameters, just use them. Note: batch size and precision use >= since
        // smaller batch sizes and/or lower precision may still use corresponding
        // cached results as reference.
        auto ref_params = test_params.make_params_for_reference_cpu();
        ref_params.validate();
        if(!ref_params.valid(verbose))
        {
            throw std::invalid_argument(
                "Reference CPU FFT parameters created from test parameters were found invalid");
        }
        // fetch from cache if applicable:
        if(cached_data.input_is_set.valid() && cached_data.output_is_set.valid()
           && cached_data.params.length == ref_params.length
           && cached_data.params.transform_type == ref_params.transform_type
           && cached_data.params.run_callbacks == ref_params.run_callbacks
           && cached_data.params.scale_factor == ref_params.scale_factor
           && cached_data.params.precision >= ref_params.precision
           && cached_data.params.nbatch >= ref_params.nbatch)
        {
            this->swap(cached_data);
            if(ref_params.precision < params.precision)
                async_narrow_precision(ref_params.precision);
            if(verbose > 3)
            {
                std::cout << "CPU params:\n";
                std::cout << params.str("\n\t") << std::endl;
            }

            return;
        }
        // cache is no longer useful, clear it and compute new one
        cached_data.clear();
        // alloc I/O buffers and create FFTW plan
        params = ref_params;
        assert(ref_params.placement == fft_placement_notinplace);
        cpu_input
            = allocate_cpu_fft_buffer(ref_params.precision, ref_params.itype, ref_params.isize);
        cpu_output
            = allocate_cpu_fft_buffer(ref_params.precision, ref_params.otype, ref_params.osize);
        switch(ref_params.precision)
        {
        case fft_precision_double:
            cpu_plan.assign(fftw_plan_via_rocfft<double>(ref_params.length,
                                                         ref_params.istride,
                                                         ref_params.ostride,
                                                         ref_params.nbatch,
                                                         ref_params.idist,
                                                         ref_params.odist,
                                                         ref_params.transform_type,
                                                         cpu_input,
                                                         cpu_output));
            break;
        case fft_precision_single:
        case fft_precision_half:
            cpu_plan.assign(fftw_plan_via_rocfft<float>(ref_params.length,
                                                        ref_params.istride,
                                                        ref_params.ostride,
                                                        ref_params.nbatch,
                                                        ref_params.idist,
                                                        ref_params.odist,
                                                        ref_params.transform_type,
                                                        cpu_input,
                                                        cpu_output));
            break;
        default:
            throw std::invalid_argument(
                "Unexpected precision encountered when constructing reference FFT data");
            break;
        }
        if(verbose > 3)
        {
            std::cout << "CPU params:\n";
            std::cout << params.str("\n\t") << std::endl;
        }
    }

    bool needs_input_initialization() const
    {
        return !input_is_set.valid();
    }

    void initialize_input(const fft_input_generator& gen)
    {
        if(!is_host_generator(gen))
            throw std::invalid_argument("Independent initialization of input data for reference "
                                        "FFT requires host-side input generator");
        params.igen  = gen; // <-- conditions the behavior of params.compute_input below
        input_is_set = std::async(std::launch::async, [&]() { params.compute_input(cpu_input); });
        // output data needs re-computing:
        invalidate_io_data(fft_io::fft_io_out);
    }

    void initialize_input_using(const std::vector<gpubuf>& src_buffers,
                                const fft_params&          test_params)
    {
        const auto ibuffer_sizes = test_params.ibuffer_sizes();
        if(can_use_direct_input_copy_with(test_params))
        {
            // Copy input to CPU
            for(unsigned int idx = 0; idx < src_buffers.size(); ++idx)
            {
                const auto hip_status = hipMemcpy(cpu_input.at(idx).data(),
                                                  src_buffers[idx].data(),
                                                  ibuffer_sizes[idx],
                                                  hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemcpy failure with error " << hip_status;
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
        else
        {
            auto tmp_host_buffers = allocate_host_buffer(ibuffer_sizes);
            // Copy input to CPU
            for(unsigned int idx = 0; idx < src_buffers.size(); ++idx)
            {
                const auto hip_status = hipMemcpy(tmp_host_buffers.at(idx).data(),
                                                  src_buffers[idx].data(),
                                                  ibuffer_sizes[idx],
                                                  hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemcpy failure with error " << hip_status;
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }

            copy_buffers(tmp_host_buffers,
                         cpu_input,
                         test_params.ilength(),
                         test_params.nbatch,
                         test_params.precision,
                         test_params.itype,
                         test_params.istride,
                         test_params.idist,
                         params.itype,
                         params.istride,
                         params.idist,
                         test_params.ioffset,
                         params.ioffset);
        }

        std::promise<void> tmp_promise;
        tmp_promise.set_value();
        input_is_set = tmp_promise.get_future();
        // output data needs re-computing:
        invalidate_io_data(fft_io::fft_io_out);
    }

    void copy_input_data_in_device_buffers(const std::vector<gpubuf>& device_input_buffers,
                                           const fft_params&          test_params) const
    {
        if(needs_input_initialization())
            throw std::logic_error("The input data of reference FFT results needs to be "
                                   "initialized before being copied");
        input_is_set.get();
        // gets a pre-computed gpu input buffer from the cpu cache
        const std::vector<hostbuf>* gpu_input = &cpu_input;
        std::vector<hostbuf>        temp_host_buffers;
        const auto                  ibuffer_sizes = test_params.ibuffer_sizes();

        if(!can_use_direct_input_copy_with(test_params))
        {
            temp_host_buffers = allocate_host_buffer(ibuffer_sizes);
            copy_buffers(cpu_input,
                         temp_host_buffers,
                         test_params.ilength(),
                         test_params.nbatch,
                         test_params.precision,
                         params.itype,
                         params.istride,
                         params.idist,
                         test_params.itype,
                         test_params.istride,
                         test_params.idist,
                         {0},
                         test_params.ioffset);
            gpu_input = &temp_host_buffers;
        }

        // Copy input to GPU
        for(unsigned int idx = 0; idx < gpu_input->size(); ++idx)
        {
            const auto hip_status = hipMemcpy(device_input_buffers[idx].data(),
                                              gpu_input->at(idx).data(),
                                              ibuffer_sizes[idx],
                                              hipMemcpyHostToDevice);

            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMemcpy failure with error " << hip_status;
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }
        }
    }

    bool needs_computing() const
    {
        return !output_is_set.valid();
    }

    ~reference_fft_data_t()
    {
        // Remember the results of the last FFT we computed with FFTW.  Tests
        // are ordered so that later cases can often reuse this result.
        if(input_is_set.valid() && output_is_set.valid() && this != &cached_data)
        {
            cached_data.swap(*this);
        }
    }
    static reference_fft_data_t make_default()
    {
        return reference_fft_data_t();
    }

    // Note: the FFTW cpu plan is destroyed when the thread completes
    void launch_async_compute()
    {
        if(!input_is_set.valid())
            throw std::logic_error("Asynchronous computation of reference FFT results mustn't be "
                                   "launche before having set the reference input data");
        output_is_set = std::async(std::launch::async, [&]() {
            input_is_set.get();
            switch(params.precision)
            {
            case(fft_precision_double):
                execute_cpu_fft<double>(
                    params, cpu_plan.get_plan_ptr<double>(), cpu_input, cpu_output);
                break;
            case(fft_precision_single):
                execute_cpu_fft<float>(
                    params, cpu_plan.get_plan_ptr<float>(), cpu_input, cpu_output);
                break;
            case(fft_precision_half):
                // cpu_input and cpu_output must be large enough for computing reference
                // results in single-precision floating-point arithmetics but the output's
                // precision is downgraded upon completion of the computation.
                execute_cpu_fft<rocfft_fp16>(
                    params, cpu_plan.get_plan_ptr<float>(), cpu_input, cpu_output);
                break;
            default:
                throw std::logic_error("Unexpected precision for the CPU plan");
            }
        });
    }

    void print_data(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("Unexpected io flag.");

        auto& flag = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!flag.valid())
            throw std::logic_error(
                "The desired reference data cannot be printed since it was not set");
        flag.get();
        if(io == fft_io::fft_io_in)
        {
            std::cout << "CPU input:\n";
            params.print_ibuffer(cpu_input);
        }
        else
        {
            std::cout << "CPU output:\n";
            params.print_obuffer(cpu_output);
        }
    }

    std::shared_future<VectorNorms> get_norm(fft_io io, size_t relevant_batch_size)
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("Unexpected io flag.");
        auto& io_is_set = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!io_is_set.valid())
        {
            throw std::logic_error(
                "Desired norm cannot be computed since corresponding data was not set");
        }
        return std::async(std::launch::async, [&, io, relevant_batch_size]() {
            io_is_set.get();
            return norm(io == fft_io::fft_io_in ? cpu_input : cpu_output,
                        io == fft_io::fft_io_in ? params.ilength() : params.olength(),
                        std::min(params.nbatch, relevant_batch_size),
                        params.precision,
                        io == fft_io::fft_io_in ? params.itype : params.otype,
                        io == fft_io::fft_io_in ? params.istride : params.ostride,
                        io == fft_io::fft_io_in ? params.idist : params.odist,
                        io == fft_io::fft_io_in ? params.ioffset : params.ooffset);
        });
    }

    const std::vector<hostbuf>& get_buffers(fft_io io)
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("Unexpected io flag.");
        auto& io_is_set = io == fft_io::fft_io_in ? input_is_set : output_is_set;
        if(!io_is_set.valid())
        {
            throw std::logic_error("Desired reference FFT results' input/output buffers cannot be "
                                   "queried since corresponding data was not set");
        }
        io_is_set.get();
        return io == fft_io::fft_io_in ? cpu_input : cpu_output;
    }
    const fft_params& get_params() const
    {
        return params;
    }

    static void clear_cache()
    {
        cached_data.clear();
    }

private:
    bool can_use_direct_input_copy_with(const fft_params& other) const
    {
        return params.itype == other.itype && params.istride == other.istride
               && params.idist == other.idist && params.isize == other.isize
               && params.precision <= other.precision;
    }

    void invalidate_io_data(fft_io io)
    {
        if(io == fft_io::fft_io_in)
        {
            input_is_set = decltype(input_is_set){};
            cpu_input.clear();
        }
        else
        {
            output_is_set = decltype(output_is_set){};
            cpu_output.clear();
        }
    }

    void clear()
    {
        params = fft_params{};
        invalidate_io_data(fft_io::fft_io_in);
        invalidate_io_data(fft_io::fft_io_out);
    }

    void swap(reference_fft_data_t& other)
    {
        if(other.input_is_set.valid())
            other.input_is_set.get();
        if(other.output_is_set.valid())
            other.output_is_set.get();

        std::swap(input_is_set, other.input_is_set);
        std::swap(output_is_set, other.output_is_set);
        cpu_input.swap(other.cpu_input);
        cpu_output.swap(other.cpu_output);
        std::swap(params, other.params);
        std::swap(cpu_plan, other.cpu_plan);
    }

    void async_narrow_precision(fft_precision narrower_prec)
    {
        const auto invalid_ref_prec_excpt = std::logic_error(
            "Invalid precision encountered for reference results to be narrowed");
        switch(narrower_prec)
        {
        case fft_precision_single:
        {
            if(params.precision != fft_precision_double)
                throw invalid_ref_prec_excpt;
            input_is_set  = std::async(std::launch::async, [&]() {
                narrow_precision_inplace<double, float>(cpu_input.front());
            });
            output_is_set = std::async(std::launch::async, [&]() {
                narrow_precision_inplace<double, float>(cpu_output.front());
            });
        }
        break;
        case fft_precision_half:
        {
            if(params.precision == fft_precision_double)
            {
                input_is_set  = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<double, rocfft_fp16>(cpu_input.front());
                });
                output_is_set = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<double, rocfft_fp16>(cpu_output.front());
                });
            }
            else if(params.precision == fft_precision_single)
            {
                input_is_set  = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<float, rocfft_fp16>(cpu_input.front());
                });
                output_is_set = std::async(std::launch::async, [&]() {
                    narrow_precision_inplace<float, rocfft_fp16>(cpu_output.front());
                });
            }
            else
                throw invalid_ref_prec_excpt;
        }
        break;
        default:
            throw std::invalid_argument(
                "Unexpected precision given to narrow reference FFT results");
        }
        params.precision = narrower_prec;
    }

    struct cpu_plan_t
    {
        cpu_plan_t()
            : precision(fft_precision_single)
            , plan_ptr{.fp32 = nullptr}
        {
        }
        // disable copies
        cpu_plan_t(const cpu_plan_t&) = delete;
        cpu_plan_t& operator=(const cpu_plan_t&) = delete;
        // default moves
        cpu_plan_t(cpu_plan_t&&) = default;
        cpu_plan_t& operator=(cpu_plan_t&&) = default;
        // assign directly from fftw plans
        void assign(const fftw_plan& raw_plan_ptr)
        {
            free();
            precision     = fft_precision_double;
            plan_ptr.fp64 = raw_plan_ptr;
        }
        void assign(const fftwf_plan& raw_plan_ptr)
        {
            free();
            precision     = fft_precision_single;
            plan_ptr.fp32 = raw_plan_ptr;
        }

        void free()
        {
            if(precision == fft_precision_single && plan_ptr.fp32)
            {
                fftwf_destroy_plan(plan_ptr.fp32);
                plan_ptr.fp32 = nullptr;
            }
            else if(precision == fft_precision_double && plan_ptr.fp64)
            {
                fftw_destroy_plan(plan_ptr.fp64);
                plan_ptr.fp64 = nullptr;
            }
        }
        ~cpu_plan_t()
        {
            free();
        }

        template <typename T>
        typename fftw_trait<T>::fftw_plan_type& get_plan_ptr()
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
            if constexpr(std::is_same_v<T, float>)
                return plan_ptr.fp32;
            else
                return plan_ptr.fp64;
        }

        fft_precision get_prec() const
        {
            return precision;
        }

    private:
        fft_precision precision;
        union
        {
            // CPU (FFTW) plans are either single or double precision
            // (no actual half precision plan on CPU)
            fftw_trait<float>::fftw_plan_type  fp32;
            fftw_trait<double>::fftw_plan_type fp64;
        } plan_ptr;
    };

    fft_params               params;
    std::shared_future<void> input_is_set;
    std::shared_future<void> output_is_set;
    // FFTW input/output
    std::vector<hostbuf> cpu_input, cpu_output;
    cpu_plan_t           cpu_plan;

    // Private default constructor. (only needed for definition of static cache)
    reference_fft_data_t() = default;

    static reference_fft_data_t cached_data;
};

// Perform several checks to make sure buffers will fit in device memory
template <class Tparams>
inline void check_problem_fits_device_memory(Tparams& params, const int verbose)
{

    int  dev_id     = hipInvalidDeviceId;
    auto hip_status = hipGetDevice(&dev_id);
    if(hip_status != hipSuccess || dev_id == hipInvalidDeviceId)
    {
        ++n_hip_failures;
        std::stringstream ss;
        ss << "hipGetDevice failed with error code " << hip_status << " reporting device ID "
           << dev_id;
        if(skip_runtime_fails)
        {
            throw ROCFFT_SKIP{ss.str()};
        }
        else
        {
            throw ROCFFT_FAIL{ss.str()};
        }
    }
    const auto vram_avail = device_memory_accountant::singleton().get_usable_bytes(dev_id);

    // First try a quick estimation of vram footprint, to speed up skipping tests
    // that are too large to fit in the gpu (no plan created with the rocFFT backend)
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);

    if(!vram_fits_problem(raw_vram_footprint, vram_avail))
    {
        std::stringstream ss;
        ss << "Raw problem size (" << byte_size_to_str(raw_vram_footprint)
           << ") exceeds usable memory on device (" << byte_size_to_str(vram_avail) << ")";
        throw ROCFFT_SKIP{ss.str()};
    }

    if(verbose > 2)
    {
        std::cout << "Raw problem size: " << raw_vram_footprint << std::endl;
    }

    // If it passed the quick estimation test, go for the more
    // accurate calculation that actually creates the plan and
    // take into account the work buffer size
    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, vram_avail))
    {
        if(verbose)
        {
            std::cout << "Problem raw data won't fit on device; skipped." << std::endl;
        }
        std::stringstream ss;
        ss << "Problem size (" << byte_size_to_str(vram_footprint)
           << " GiB) exceeds usable memory on device (" << byte_size_to_str(vram_avail) << ")";
        throw ROCFFT_SKIP{ss.str()};
    }
}

template <typename Tfloat>
bool fftw_plan_uses_bluestein(const typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan)
{
#ifdef FFTW_HAVE_SPRINT_PLAN
    char*       print_plan_c_str = fftw_sprint_plan<Tfloat>(cpu_plan);
    std::string print_plan(print_plan_c_str);
    free(print_plan_c_str);
    return print_plan.find("bluestein") != std::string::npos;
#else
    // assume worst case (bluestein is always used)
    return true;
#endif
}

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        return info.param.token();
    }
};

// execute the GPU transform
template <class Tparams>
inline void execute_gpu_fft(Tparams&              params,
                            std::vector<void*>&   pibuffer,
                            std::vector<void*>&   pobuffer,
                            std::vector<gpubuf>&  obuffer,
                            std::vector<hostbuf>& gpu_output,
                            bool                  round_trip_inverse = false)
{
    // Vector of callback data - at function scope so they live until
    // after the transform is completed
    std::vector<gpubuf_t<callback_test_data>> all_cb_data;

    std::vector<void*> load_cb_func;
    std::vector<void*> load_cb_data;
    std::vector<void*> store_cb_func;
    std::vector<void*> store_cb_data;

    if(params.run_callbacks)
    {
        auto runtime_err_handler = [&](const std::string& msg) {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{msg};
            }
            else
            {
                throw ROCFFT_FAIL{msg};
            }
        };

        get_rank_load_callbacks(params,
                                load_cb_func,
                                load_cb_data,
                                runtime_err_handler,
                                round_trip_inverse,
                                all_cb_data);
        get_rank_store_callbacks(params,
                                 store_cb_func,
                                 store_cb_data,
                                 runtime_err_handler,
                                 round_trip_inverse,
                                 all_cb_data);

        auto fft_status
            = params.set_callbacks(&load_cb_func, &load_cb_data, &store_cb_func, &store_cb_data);
        if(fft_status != fft_status_success)
            throw std::runtime_error("set callback failure");
    }

    // Execute the transform:
    auto fft_status = params.execute(pibuffer.data(), pobuffer.data());
    if(fft_status != fft_status_success)
        throw std::runtime_error("FFT plan execution failure");

    // if not comparing, then just executing the GPU FFT is all we
    // need to do
    if(!fftw_compare)
        return;

    ASSERT_TRUE(!gpu_output.empty()) << "no output buffers";

    // if output is in multiple bricks, collect it into the
    // host buffer where the results need to go
    params.multi_gpu_finalize(gpu_output, obuffer, pobuffer);

    if(params.ofields.empty())
    {
        // otherwise, copy directly from the device
        for(unsigned int idx = 0; idx < gpu_output.size(); ++idx)
        {
            ASSERT_TRUE(gpu_output[idx].data() != nullptr)
                << "output buffer index " << idx << " is empty";
            auto hip_status = hipMemcpy(gpu_output[idx].data(),
                                        pobuffer.at(idx),
                                        gpu_output[idx].size(),
                                        hipMemcpyDeviceToHost);
            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMemcpy failure";
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }
        }
    }
    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        params.print_obuffer(gpu_output);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        params.print_obuffer_flat(gpu_output);
    }
}

template <typename Tfloat>
static void assert_init_value(const std::vector<hostbuf>& output,
                              const size_t                idx,
                              const Tfloat                orig_value);

template <>
void assert_init_value(const std::vector<hostbuf>& output, const size_t idx, const float orig_value)
{
    float actual_value = reinterpret_cast<const float*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const double                orig_value)
{
    double actual_value = reinterpret_cast<const double*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const rocfft_complex<float> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<float> actual_value
            = reinterpret_cast<const rocfft_complex<float>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<float> actual_value{
            reinterpret_cast<const float*>(output.front().data())[idx],
            reinterpret_cast<const float*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

template <>
void assert_init_value(const std::vector<hostbuf>&  output,
                       const size_t                 idx,
                       const rocfft_complex<double> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<double> actual_value
            = reinterpret_cast<const rocfft_complex<double>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<double> actual_value{
            reinterpret_cast<const double*>(output.front().data())[idx],
            reinterpret_cast<const double*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

static const int OUTPUT_INIT_PATTERN = 0xcd;
template <class Tfloat>
void check_single_output_stride(const std::vector<hostbuf>& output,
                                const size_t                offset,
                                const std::vector<size_t>&  length,
                                const std::vector<size_t>&  stride,
                                const size_t                i)
{
    Tfloat orig;
    memset(static_cast<void*>(&orig), OUTPUT_INIT_PATTERN, sizeof(Tfloat));

    size_t curLength         = length[i];
    size_t curStride         = stride[i];
    size_t nextSmallerLength = i == length.size() - 1 ? 0 : length[i + 1];
    size_t nextSmallerStride = i == stride.size() - 1 ? 0 : stride[i + 1];

    if(nextSmallerLength == 0)
    {
        // this is the fastest dim, indexes that are not multiples of
        // the stride should be the initial value
        for(size_t idx = 0; idx < (curLength - 1) * curStride; ++idx)
        {
            if(idx % curStride != 0)
                assert_init_value<Tfloat>(output, idx, orig);
        }
    }
    else
    {
        for(size_t lengthIdx = 0; lengthIdx < curLength; ++lengthIdx)
        {
            // check that the space after the next smaller dim and the
            // end of this dim is initial value
            for(size_t idx = nextSmallerLength * nextSmallerStride; idx < curStride; ++idx)
                assert_init_value<Tfloat>(output, idx, orig);

            check_single_output_stride<Tfloat>(
                output, offset + lengthIdx * curStride, length, stride, i + 1);
        }
    }
}

template <class Tparams>
void check_output_strides(const std::vector<hostbuf>& output, Tparams& params)
{
    // treat batch+dist like highest length+stride, if batch > 1
    std::vector<size_t> length;
    std::vector<size_t> stride;
    if(params.nbatch > 1)
    {
        length.push_back(params.nbatch);
        stride.push_back(params.odist);
    }

    auto olength = params.olength();
    std::copy(olength.begin(), olength.end(), std::back_inserter(length));
    std::copy(params.ostride.begin(), params.ostride.end(), std::back_inserter(stride));

    if(params.precision == fft_precision_single)
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<float>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<float>>(output, 0, length, stride, 0);
    }
    else
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<double>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<double>>(output, 0, length, stride, 0);
    }
}

// run rocFFT inverse transform
template <class Tparams>
inline void run_round_trip_inverse(Tparams&              params,
                                   std::vector<gpubuf>&  obuffer,
                                   std::vector<void*>&   pibuffer,
                                   std::vector<void*>&   pobuffer,
                                   std::vector<hostbuf>& gpu_output)
{
    params.validate();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    // Create FFT plan - this will also allocate work buffer, but will throw a
    // specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        std::stringstream ss;
        ss << "Failed to allocate work buffer (size: " << e.attempted_size << ")";
        ++n_hip_failures;
        if(skip_runtime_fails)
        {
            throw ROCFFT_SKIP{ss.str()};
        }
        else
        {
            throw ROCFFT_FAIL{ss.str()};
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "round trip inverse plan creation failed";

    auto obuffer_sizes = params.obuffer_sizes();

    if(params.placement != fft_placement_inplace)
    {
        for(unsigned int i = 0; i < obuffer_sizes.size(); ++i)
        {
            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                auto hip_status = hipMemset(pobuffer[i], OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemset failure";
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
    }

    if(params.multiGPU > 1)
    {
        if(verbose > 0)
        {
            std::cout << "scattering data for multi-GPU inverse" << std::endl;
        }
        std::vector<hostbuf> cpu_input;
        params.multi_gpu_prepare(cpu_input, obuffer, pibuffer, pobuffer);
    }

    // execute GPU transform
    execute_gpu_fft(params, pibuffer, pobuffer, obuffer, gpu_output, true);
}

// compare rocFFT inverse transform with forward transform input
template <class Tparams>
inline void compare_round_trip_inverse(Tparams&                    params,
                                       const fft_params&           contiguous_params,
                                       std::vector<hostbuf>&       gpu_output,
                                       const std::vector<hostbuf>& cpu_input,
                                       const VectorNorms&          cpu_input_norm,
                                       size_t                      total_length)
{
    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm = std::async(std::launch::async, [&]() {
        return norm(gpu_output,
                    params.olength(),
                    params.nbatch,
                    params.precision,
                    params.otype,
                    params.ostride,
                    params.odist,
                    params.ooffset);
    });

    // compare GPU inverse output to CPU forward input
    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    const double linf_cutoff
        = type_epsilon(params.precision) * cpu_input_norm.l_inf * log(total_length);

    VectorNorms diff = distance(cpu_input,
                                gpu_output,
                                params.olength(),
                                params.nbatch,
                                params.precision,
                                contiguous_params.itype,
                                contiguous_params.istride,
                                contiguous_params.idist,
                                params.otype,
                                params.ostride,
                                params.odist,
                                linf_failures.get(),
                                linf_cutoff,
                                {0},
                                params.ooffset,
                                1.0 / total_length);

    if(verbose > 1)
    {
        std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures->begin(), linf_failures->end());
        for(const auto& i : *linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half
            = std::max(max_linf_eps_half, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_half
            = std::max(max_l2_eps_half, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_single
            = std::max(max_l2_eps_single, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_double
            = std::max(max_l2_eps_double, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    EXPECT_TRUE(diff.l_inf <= linf_cutoff)
        << "Linf test failed.  Linf:" << diff.l_inf
        << "\tnormalized Linf: " << diff.l_inf / cpu_input_norm.l_inf << "\tcutoff: " << linf_cutoff
        << params.str();

    EXPECT_TRUE(diff.l_2 / cpu_input_norm.l_2
                <= sqrt(log2(total_length)) * type_epsilon(params.precision))
        << "L2 test failed. L2: " << diff.l_2
        << "\tnormalized L2: " << diff.l_2 / cpu_input_norm.l_2
        << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
        << params.str();
}

static void allocate_device_input_buffers() {}

// run CPU + rocFFT transform with the given params and compare
template <class Tfloat, class Tparams>
inline void fft_vs_reference_impl(Tparams& params, bool round_trip)
{
    // Call hipGetLastError to reset any errors
    // returned by previous HIP runtime API calls.
    hipError_t hip_status = hipGetLastError();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    // Make sure FFT buffers fit in device memory
    check_problem_fits_device_memory(params, verbose);
    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();

    // Allocate device input buffer(s)
    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            std::stringstream ss;
            if(hip_status == hipErrorOutOfMemory)
            {
                ss << "Input buffer size (" << bytes_to_GiB(ibuffer_sizes[i])
                   << " GiB) raw data too large for device";
            }
            else
            {
                ss << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i]
                   << "(" << bytes_to_GiB(ibuffer_sizes[i]) << " GiB)"
                   << " with code " << hipError_to_string(hip_status);
            }
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                throw ROCFFT_SKIP{ss.str()};
            }
            else
            {
                throw ROCFFT_FAIL{ss.str()};
            }
        }
        pibuffer[i] = ibuffer[i].data();
    }

    // Construct structure of feference FFTW results if needed
    std::optional<reference_fft_data_t> reference_results;
    if(fftw_compare)
    {
        reference_results.emplace(params);
        if(reference_results->needs_input_initialization() && is_host_generator(params.igen))
            reference_results->initialize_input(params.igen);
    }
    // Initialize input data buffer(s) on device
    if(reference_results && !reference_results->needs_input_initialization())
    {
        reference_results->copy_input_data_in_device_buffers(ibuffer, params);
    }
    else
    {
        // Input data buffer(s) on device may be initialized independently of
        // reference results
        if(is_device_generator(params.igen))
            params.compute_input(ibuffer);
        else
        {
            auto tmp_host_buffers = allocate_host_buffer(ibuffer_sizes);
            params.compute_input(tmp_host_buffers);
            for(unsigned int idx = 0; idx < tmp_host_buffers.size(); ++idx)
            {
                hip_status = hipMemcpy(ibuffer[idx].data(),
                                       tmp_host_buffers.at(idx).data(),
                                       ibuffer_sizes[idx],
                                       hipMemcpyHostToDevice);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemcpy failure with error " << hip_status;
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
    }

    if(reference_results)
    {
        if(reference_results->needs_input_initialization())
            reference_results->initialize_input_using(ibuffer, params);
        if(reference_results->needs_computing())
            reference_results->launch_async_compute();
    }

    // Create FFT plan - this will also allocate work buffer, but
    // will throw a specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        ++n_hip_failures;
        std::stringstream ss;
        ss << "Work buffer allocation failed with size: " << e.attempted_size;
        if(skip_runtime_fails)
        {
            throw ROCFFT_SKIP{ss.str()};
        }
        else
        {
            throw ROCFFT_FAIL{ss.str()};
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "plan creation failed";

    if(verbose > 3 && reference_results)
        reference_results->print_data(fft_io::fft_io_in);

    // compute input norm
    std::shared_future<VectorNorms> cpu_input_norm;
    if(reference_results)
        cpu_input_norm = reference_results->get_norm(fft_io::fft_io_in, params.nbatch);

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    std::vector<void*>   pobuffer;

    // allocate the output buffer

    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                   << "(" << bytes_to_GiB(obuffer_sizes[i]) << " GiB)"
                   << " with code " << hipError_to_string(hip_status);
                if(skip_runtime_fails)
                {
                    throw ROCFFT_SKIP{ss.str()};
                }
                else
                {
                    throw ROCFFT_FAIL{ss.str()};
                }
            }

            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                hip_status
                    = hipMemset(obuffer_data[i].data(), OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    std::stringstream ss;
                    ss << "hipMemset failure with error " << hip_status;
                    if(skip_runtime_fails)
                    {
                        throw ROCFFT_SKIP{ss.str()};
                    }
                    else
                    {
                        throw ROCFFT_FAIL{ss.str()};
                    }
                }
            }
        }
    }
    pobuffer.resize(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }
    // scatter data out to multi-GPUs if this is a multi-GPU test
    if(reference_results)
        params.multi_gpu_prepare(
            reference_results->get_buffers(fft_io::fft_io_in), ibuffer, pibuffer, pobuffer);

    // Run CPU transform
    //
    // NOTE: This must happen after input is copied to GPU and input
    // norm is computed, since the CPU FFT may overwrite the input.
    std::shared_future<VectorNorms> cpu_output_norm;
    if(reference_results)
    {
        if(verbose > 3)
            reference_results->print_data(fft_io::fft_io_out);
        cpu_output_norm = reference_results->get_norm(fft_io::fft_io_out, params.nbatch);
    }

    // execute GPU transform
    std::vector<hostbuf> gpu_output
        = allocate_host_buffer(params.precision, params.otype, params.osize);

    execute_gpu_fft(params, pibuffer, pobuffer, *obuffer, gpu_output);

    params.free();

    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm;
    if(fftw_compare)
        gpu_norm = std::async(std::launch::async, [&]() {
            return norm(gpu_output,
                        params.olength(),
                        params.nbatch,
                        params.precision,
                        params.otype,
                        params.ostride,
                        params.odist,
                        params.ooffset);
        });

    // compare output
    //
    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    // wait for cpu FFT so we can compute cutoff

    const auto total_length = product(params.length.begin(), params.length.end());

    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    double      linf_cutoff;
    VectorNorms diff;

    std::shared_future<void> compare_output;
    if(fftw_compare)
        compare_output = std::async(std::launch::async, [&]() {
            linf_cutoff
                = type_epsilon(params.precision) * cpu_output_norm.get().l_inf * log(total_length);

            diff = distance(reference_results->get_buffers(fft_io::fft_io_out),
                            gpu_output,
                            params.olength(),
                            params.nbatch,
                            params.precision,
                            reference_results->get_params().otype,
                            reference_results->get_params().ostride,
                            reference_results->get_params().odist,
                            params.otype,
                            params.ostride,
                            params.odist,
                            linf_failures.get(),
                            linf_cutoff,
                            {0},
                            params.ooffset);
        });

    if(compare_output.valid())
        compare_output.get();

    Tparams              params_inverse;
    std::vector<hostbuf> rountrip_output_buffers;

    if(round_trip)
    {
        params_inverse.inverse_from_forward(params);
        params_inverse.compute_osize();
        rountrip_output_buffers = allocate_host_buffer(params_inverse.obuffer_sizes());

        run_round_trip_inverse<Tparams>(
            params_inverse, *obuffer, pobuffer, pibuffer, rountrip_output_buffers);
    }

    if(fftw_compare)
    {
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_2));
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_inf));

        ASSERT_TRUE(std::isfinite(cpu_output_norm.get().l_2));
        ASSERT_TRUE(std::isfinite(cpu_output_norm.get().l_inf));

        if(verbose > 1)
        {
            std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
            std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
            std::cout << "GPU linf norm failures:";
            std::sort(linf_failures->begin(), linf_failures->end());
            for(const auto& i : *linf_failures)
            {
                std::cout << " (" << i.first << "," << i.second << ")";
            }
            std::cout << std::endl;
        }

        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();
    }

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half = std::max(max_linf_eps_half,
                                     diff.l_inf / cpu_output_norm.get().l_inf / log(total_length));
        max_l2_eps_half   = std::max(max_l2_eps_half,
                                   diff.l_2 / cpu_output_norm.get().l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single = std::max(
            max_linf_eps_single, diff.l_inf / cpu_output_norm.get().l_inf / log(total_length));
        max_l2_eps_single = std::max(
            max_l2_eps_single, diff.l_2 / cpu_output_norm.get().l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double = std::max(
            max_linf_eps_double, diff.l_inf / cpu_output_norm.get().l_inf / log(total_length));
        max_l2_eps_double = std::max(
            max_l2_eps_double, diff.l_2 / cpu_output_norm.get().l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    if(fftw_compare)
    {
        EXPECT_TRUE(diff.l_inf <= linf_cutoff)
            << "Linf test failed.  Linf:" << diff.l_inf
            << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.get().l_inf
            << "\tcutoff: " << linf_cutoff << "\n"
            << params.str();

        EXPECT_TRUE(diff.l_2 / cpu_output_norm.get().l_2
                    <= sqrt(log2(total_length)) * type_epsilon(params.precision))
            << "L2 test failed. L2: " << diff.l_2
            << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.get().l_2
            << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision) << "\n"
            << params.str();
    }

    if(round_trip && fftw_compare)
    {
        compare_round_trip_inverse<Tparams>(params_inverse,
                                            reference_results->get_params(),
                                            rountrip_output_buffers,
                                            reference_results->get_buffers(fft_io::fft_io_in),
                                            cpu_input_norm.get(),
                                            total_length);
    }
}

#endif
