#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <vector>


torch::Tensor get_bbox_from_mask_cpp(
    const torch::Tensor &mask,
    int64_t size_divisor = 1
) {
    // Basic input validation
    TORCH_CHECK(mask.dim() >= 2, "Input tensor must have at least 2 dimensions");
    const auto orig_shape = mask.sizes();
    const int64_t num_dims = orig_shape.size();

    // Calculate sample count from leading dimensions
    int64_t num_samples = 1;
    for (int i = 0; i < num_dims - 2; ++i) {
        num_samples *= orig_shape[i];
    }
    const int64_t H = orig_shape[num_dims - 2];
    const int64_t W = orig_shape[num_dims - 1];

    // Reshape to 3D tensor (samples, H, W) with contiguous memory
    auto mask_flat = mask.view({num_samples, H, W}).contiguous();

    // Create output tensor with proper device
    auto options = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(mask.device());
    auto bbox = torch::zeros({num_samples, 4}, options);

    // Get device type for conditional logic
    const bool is_cuda = mask.is_cuda();

    // Parallel processing using PyTorch's API
    at::parallel_for(0, num_samples, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            auto sample = mask_flat[i];

            // Process on CPU to avoid device sync issues
            auto cpu_sample = is_cuda ? sample.cpu() : sample;

            // Find valid rows/columns
            auto rows = cpu_sample.any(1).squeeze(-1);
            auto cols = cpu_sample.any(0).squeeze(-1);

            // Find row boundaries
            auto row_indices = torch::nonzero(rows);
            bool found_row = row_indices.size(0) > 0;
            int64_t up_row = 0, bottom_row = H-1;
            if (found_row) {
                up_row = row_indices[0][0].item<int64_t>();
                bottom_row = row_indices[-1][0].item<int64_t>();
            }

            // Find column boundaries
            auto col_indices = torch::nonzero(cols);
            bool found_col = col_indices.size(0) > 0;
            int64_t left_col = 0, right_col = W-1;
            if (found_col) {
                left_col = col_indices[0][0].item<int64_t>();
                right_col = col_indices[-1][0].item<int64_t>();
            }

            // Calculate coordinates
            int64_t coords[4];
            if (found_row && found_col) {
                coords[0] = left_col;
                coords[1] = up_row;
                coords[2] = right_col + 1;
                coords[3] = bottom_row + 1;
            } else {
                coords[0] = 0;
                coords[1] = 0;
                coords[2] = W;
                coords[3] = H;
            }

            // Apply size divisor with clamping
            coords[0] = std::clamp((coords[0]/size_divisor)*size_divisor, 0L, W);
            coords[1] = std::clamp((coords[1]/size_divisor)*size_divisor, 0L, H);
            coords[2] = std::clamp(
                ((coords[2]+size_divisor-1)/size_divisor)*size_divisor,
                coords[0], W
            );
            coords[3] = std::clamp(
                ((coords[3]+size_divisor-1)/size_divisor)*size_divisor,
                coords[1], H
            );

            // Use PyTorch's device-aware assignment
            if (is_cuda) {
                bbox[i][0] = coords[0];
                bbox[i][1] = coords[1];
                bbox[i][2] = coords[2];
                bbox[i][3] = coords[3];
            } else {
                auto bbox_ptr = bbox[i].data_ptr<int64_t>();
                std::copy(coords, coords + 4, bbox_ptr);
            }
        }
    });

    // Reshape to original dimensions
    std::vector<int64_t> new_shape;
    for (int i = 0; i < num_dims - 2; ++i) {
        new_shape.push_back(orig_shape[i]);
    }
    new_shape.push_back(4);
    return bbox.view(new_shape);
}


torch::Tensor convert_bbox_to_mask_cpp(
    const torch::Tensor& bbox,
    std::pair<int64_t, int64_t> size) {

    // Validate input dimensions
    TORCH_CHECK(bbox.size(-1) == 4,
        "Last dimension of bbox must be 4, got ", bbox.size(-1));
    TORCH_CHECK(size.first > 0 && size.second > 0,
        "Invalid mask size: ", size.first, "x", size.second);

    // Get original shape and flatten to (N, 4)
    auto orig_shape = bbox.sizes().vec();
    orig_shape.pop_back();
    auto flat_bbox = bbox.view({-1, 4});
    const int64_t num_boxes = flat_bbox.size(0);
    const auto [H, W] = size;

    // Create output tensor with proper device and dtype
    auto options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(bbox.device());
    auto mask = torch::zeros({num_boxes, H, W}, options);

    // Parallel processing using PyTorch's parallel_for
    at::parallel_for(0, num_boxes, 0, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            auto box = flat_bbox[i];
            int64_t left = box[0].item<int64_t>();
            int64_t top = box[1].item<int64_t>();
            int64_t right = box[2].item<int64_t>();
            int64_t bottom = box[3].item<int64_t>();

            // Clamp coordinates to valid range
            left = std::clamp(left, 0L, W);
            right = std::clamp(right, left, W);
            top = std::clamp(top, 0L, H);
            bottom = std::clamp(bottom, top, H);

            if (right > left && bottom > top) {
                mask[i].slice(0, top, bottom).slice(1, left, right).fill_(1);
            }
        }
    });

    // Reshape to original dimensions + (H, W)
    std::vector<int64_t> new_shape(orig_shape);
    new_shape.insert(new_shape.end(), {H, W});
    return mask.view(new_shape);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_bbox_from_mask",
          &get_bbox_from_mask_cpp,
          "Convert masks to bounding boxes with size divisor support");

    m.def("convert_bbox_to_mask",
          &convert_bbox_to_mask_cpp,
          "Convert bounding boxes to binary masks with GPU support");
}
