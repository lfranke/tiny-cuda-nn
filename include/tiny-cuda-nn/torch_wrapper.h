/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *//*
*/

/** @file   torch_bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */
#pragma once

#include <ATen/cuda/CUDAUtils.h>

#ifdef snprintf
#    undef snprintf
#endif


#include <tiny-cuda-nn/cpp_api.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x)                                                                  \
    do                                                                                  \
    {                                                                                   \
        if (!(x)) throw std::runtime_error(std::string(FILE_LINE " check failed " #x)); \
    } while (0)

inline c10::ScalarType torch_type(tcnn::cpp::EPrecision precision)
{
    switch (precision)
    {
        case tcnn::cpp::EPrecision::Fp32:
            return torch::kFloat32;
        case tcnn::cpp::EPrecision::Fp16:
            return torch::kHalf;
        default:
            throw std::runtime_error{"Unknown precision tcnn->torch"};
    }
}

inline void* void_data_ptr(torch::Tensor& tensor)
{
    switch (tensor.scalar_type())
    {
        case torch::kFloat32:
            return tensor.data_ptr<float>();
        case torch::kHalf:
            return tensor.data_ptr<torch::Half>();
        default:
            throw std::runtime_error{"Unknown precision torch->void"};
    }
}

class TorchTcnnWrapperModule
{
   public:
    TorchTcnnWrapperModule() {}
    TorchTcnnWrapperModule(tcnn::cpp::Module* module) : m_module{module} {}

    std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input, torch::Tensor params)
    {
        // Types
        CHECK_THROW(input.scalar_type() == torch::kFloat32);
        CHECK_THROW(params.scalar_type() == c10_param_precision());

        // Sizes
        std::cout << input.size(1) << " -- " << n_input_dims() << std::endl;
        CHECK_THROW(input.size(1) == n_input_dims());
        CHECK_THROW(params.size(0) == n_params());

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        uint32_t batch_size  = input.size(0);
        torch::Tensor output = torch::empty({batch_size, n_output_dims()},
                                            torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));

        tcnn::cpp::Context ctx;
        if (!input.requires_grad() && !params.requires_grad())
        {
            m_module->inference(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output),
                                void_data_ptr(params));
        }
        else
        {
            ctx = m_module->forward(stream, batch_size, input.data_ptr<float>(), void_data_ptr(output),
                                    void_data_ptr(params), input.requires_grad());
        }

        return {std::move(ctx), output};
    }

    std::tuple<torch::Tensor, torch::Tensor> bwd(const tcnn::cpp::Context& ctx, torch::Tensor input,
                                                 torch::Tensor params, torch::Tensor output, torch::Tensor dL_doutput)
    {
        if (!ctx.ctx)
        {
            throw std::runtime_error{
                "Module::bwd: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
        }

        // Types
        CHECK_THROW(input.scalar_type() == torch::kFloat32);
        CHECK_THROW(params.scalar_type() == c10_param_precision());
        CHECK_THROW(output.scalar_type() == c10_output_precision());
        CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

        // Sizes
        CHECK_THROW(input.size(1) == n_input_dims());
        CHECK_THROW(output.size(1) == n_output_dims());
        CHECK_THROW(params.size(0) == n_params());
        CHECK_THROW(output.size(0) == input.size(0));
        CHECK_THROW(dL_doutput.size(0) == input.size(0));

        uint32_t batch_size = input.size(0);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        torch::Tensor dL_dinput;
        if (input.requires_grad())
        {
            dL_dinput = torch::empty({batch_size, input.size(1)},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }

        torch::Tensor dL_dparams;
        if (params.requires_grad())
        {
            dL_dparams =
                torch::empty({n_params()}, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
        }

        if (input.requires_grad() || params.requires_grad())
        {
            m_module->backward(stream, ctx, batch_size, input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr,
                               void_data_ptr(dL_doutput), params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
                               input.data_ptr<float>(), void_data_ptr(output), void_data_ptr(params));
        }

        return {dL_dinput, dL_dparams};
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bwd_bwd_input(const tcnn::cpp::Context& ctx,
                                                                          torch::Tensor input, torch::Tensor params,
                                                                          torch::Tensor dL_ddLdinput,
                                                                          torch::Tensor dL_doutput)
    {
        // from: dL_ddLdinput
        // to:   dL_ddLdoutput, dL_dparams, dL_dinput

        if (!ctx.ctx)
        {
            throw std::runtime_error{
                "Module::bwd_bwd_input: called with invalid context. fwd likely (mistakenly) ran in inference mode."};
        }

        // Types
        CHECK_THROW(input.scalar_type() == torch::kFloat32);
        CHECK_THROW(dL_ddLdinput.scalar_type() == torch::kFloat32);
        CHECK_THROW(params.scalar_type() == c10_param_precision());
        CHECK_THROW(dL_doutput.scalar_type() == c10_output_precision());

        // Sizes
        CHECK_THROW(input.size(1) == n_input_dims());
        CHECK_THROW(dL_doutput.size(1) == n_output_dims());
        CHECK_THROW(dL_ddLdinput.size(1) == n_input_dims());
        CHECK_THROW(params.size(0) == n_params());
        CHECK_THROW(dL_doutput.size(0) == input.size(0));
        CHECK_THROW(dL_ddLdinput.size(0) == input.size(0));

        uint32_t batch_size = input.size(0);

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        torch::Tensor dL_ddLdoutput;
        if (dL_doutput.requires_grad())
        {
            dL_ddLdoutput = torch::zeros({batch_size, n_output_dims()},
                                         torch::TensorOptions().dtype(c10_output_precision()).device(torch::kCUDA));
        }

        torch::Tensor dL_dparams;
        if (params.requires_grad())
        {
            dL_dparams =
                torch::zeros({n_params()}, torch::TensorOptions().dtype(c10_param_precision()).device(torch::kCUDA));
        }

        torch::Tensor dL_dinput;
        if (input.requires_grad())
        {
            dL_dinput = torch::zeros({batch_size, n_input_dims()},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        }

        if (dL_doutput.requires_grad() || params.requires_grad())
        {
            m_module->backward_backward_input(
                stream, ctx, batch_size, dL_ddLdinput.data_ptr<float>(), input.data_ptr<float>(),
                (params.requires_grad() || input.requires_grad()) ? void_data_ptr(dL_doutput) : nullptr,
                params.requires_grad() ? void_data_ptr(dL_dparams) : nullptr,
                dL_doutput.requires_grad() ? void_data_ptr(dL_ddLdoutput) : nullptr,
                input.requires_grad() ? dL_dinput.data_ptr<float>() : nullptr, void_data_ptr(params));
        }

        return {dL_ddLdoutput, dL_dparams, dL_dinput};
    }

    torch::Tensor initial_params(size_t seed)
    {
        torch::Tensor output =
            torch::zeros({n_params()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        m_module->initialize_params(seed, output.data_ptr<float>());
        return output.to(c10_param_precision());
    }

    uint32_t n_input_dims() const { return m_module->n_input_dims(); }

    uint32_t n_params() const { return (uint32_t)m_module->n_params(); }

    tcnn::cpp::EPrecision param_precision() const { return m_module->param_precision(); }

    c10::ScalarType c10_param_precision() const { return torch_type(param_precision()); }

    uint32_t n_output_dims() const { return m_module->n_output_dims(); }

    tcnn::cpp::EPrecision output_precision() const { return m_module->output_precision(); }

    c10::ScalarType c10_output_precision() const { return torch_type(output_precision()); }

    nlohmann::json hyperparams() const { return m_module->hyperparams(); }

    std::string name() const { return m_module->name(); }

   private:
    std::unique_ptr<tcnn::cpp::Module> m_module;
};

// #if !defined(TCNN_NO_NETWORKS)
//  Module create_network_with_input_encoding(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json&
//  encoding, const nlohmann::json& network) {
//    return Module{tcnn::cpp::create_network_with_input_encoding(n_input_dims, n_output_dims, encoding, network)};
// }
//
//  Module create_network(uint32_t n_input_dims, uint32_t n_output_dims, const nlohmann::json& network) {
//    return Module{tcnn::cpp::create_network(n_input_dims, n_output_dims, network)};
// }
// #endif
//
//  Module create_encoding(uint32_t n_input_dims, const nlohmann::json& encoding, tcnn::cpp::EPrecision
//  requested_precision) {
//    return Module{tcnn::cpp::create_encoding(n_input_dims, encoding, requested_precision)};
// }


class TnnInfo : public torch::CustomClassHolder
{
   public:
    TorchTcnnWrapperModule* module = nullptr;
    tcnn::cpp::Context native_ctx;
};

// TORCH_LIBRARY(asdfasdf, m)
//{
//     std::cout << "register TnnInfo" << std::endl;
//     m.class_<TnnInfo>("TnnInfo").def(torch::init());
// }

namespace torch::autograd
{
struct _moduleFunction : public Function<_moduleFunction>
{
    // returns a tensor for every layer
    static std::vector<torch::Tensor> forward(AutogradContext* ctx, IValue native_tcnn_module, torch::Tensor input,
                                              torch::Tensor params, IValue loss_scale)
    {
        // If no output gradient is provided, no need to
        // automatically materialize it as torch.zeros.
        ctx->set_materialize_grads(false);

        TnnInfo* info = native_tcnn_module.toCustomClass<TnnInfo>().get();
        CHECK_NOTNULL(info->module);
        auto [native_ctx, output] = info->module->fwd(input, params);

        variable_list to_save;
        to_save.push_back(input);
        to_save.push_back(params);
        to_save.push_back(output);
        ctx->save_for_backward(to_save);

        info->native_ctx                      = std::move(native_ctx);
        ctx->saved_data["native_tcnn_module"] = native_tcnn_module;
        ctx->saved_data["loss_scale"]         = loss_scale;

        std::vector<torch::Tensor> output_vec;
        output_vec.push_back(output);
        return output_vec;
    }

    static std::vector<torch::Tensor> backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output)
    {
        CHECK_EQ(grad_output.size(), 1);
        auto doutput = grad_output.front();

        auto saved_tensors = ctx->get_saved_variables();
        auto input         = saved_tensors[0];
        auto params        = saved_tensors[1];
        auto output        = saved_tensors[2];

        TnnInfo* info = ctx->saved_data["native_tcnn_module"].toCustomClass<TnnInfo>().get();
        CHECK_NOTNULL(info->module);


        {
            torch::NoGradGuard ngg;

            // float loss_scale = ctx->saved_data["loss_scale"].toDouble();
            // auto scaled_grad               = doutput * loss_scale;
            auto scaled_grad               = doutput;
            auto [input_grad, weight_grad] = info->module->bwd(info->native_ctx, input, params, output, scaled_grad);

            // if (input_grad.defined())
            // {
            //     input_grad = input_grad / loss_scale;
            // }
            // if (weight_grad.defined())
            // {
            //     weight_grad = weight_grad / loss_scale;
            // }

            std::vector<torch::Tensor> output_vec;
            output_vec.push_back({});
            output_vec.push_back(input_grad);
            output_vec.push_back(weight_grad);
            output_vec.push_back({});
            return output_vec;
        }
    }
};
}  // namespace torch::autograd


class TcnnTorchModuleImpl : public torch::nn::Module
{
   public:
    TcnnTorchModuleImpl(TorchTcnnWrapperModule _module, int seed = 1337) : module(std::move(_module))
    {
        auto initial_params = module.initial_params(seed);
        params              = initial_params;
        register_parameter("params", params);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto info_unq    = std::make_unique<TnnInfo>();
        info_unq->module = &module;
        c10::intrusive_ptr<TnnInfo> info_ptr(std::move(info_unq));
        torch::IValue val(std::move(info_ptr));
        std::vector<torch::Tensor> result = torch::autograd::_moduleFunction::apply(val, x, params, loss_scale);
        CHECK_EQ(result.size(), 1);
        return result.front();
    }

    TorchTcnnWrapperModule module;
    torch::Tensor params;
    float loss_scale = 1;
};


TORCH_MODULE(TcnnTorchModule);
