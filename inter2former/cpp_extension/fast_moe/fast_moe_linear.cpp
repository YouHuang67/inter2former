#include <torch/extension.h>
#include <ATen/Parallel.h>

torch::Tensor moe_linear_forward_sorted(
    torch::Tensor input_sorted,
    torch::Tensor expert_offsets,
    torch::Tensor weights,
    torch::Tensor biases) {

    torch::NoGradGuard no_grad;

    const int64_t num_experts = expert_offsets.size(0) - 1;
    const int64_t out_features = weights.size(1);

    auto options = torch::TensorOptions()
        .dtype(input_sorted.dtype())
        .device(input_sorted.device())
        .requires_grad(false);

    auto output = torch::empty({input_sorted.size(0), out_features}, options);

    at::parallel_for(0, num_experts, 0, [&](int64_t start, int64_t end){
        for(int64_t e = start; e < end; ++e){
            const int64_t token_start = expert_offsets[e].item<int64_t>();
            const int64_t token_end = expert_offsets[e+1].item<int64_t>();
            if(token_start >= token_end) continue;

            auto expert_input = input_sorted.slice(0, token_start, token_end);
            auto expert_output = output.slice(0, token_start, token_end);

            torch::NoGradGuard no_grad_local;
            torch::addmm_out(
                expert_output,
                biases[e],
                expert_input,
                weights[e].transpose(0, 1)
            );
        }
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_sorted", &moe_linear_forward_sorted, "Optimized MoE Linear forward");
}
