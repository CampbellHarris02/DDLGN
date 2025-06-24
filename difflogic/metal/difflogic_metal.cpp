// difflogic_metal.cpp (FIXED V5)

#include <torch/extension.h>
// Essential headers for ATen types and dispatch macros
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>
#include <c10/util/TypeCast.h>

#include <cmath>
#include <array>
#include <vector>
#include <stdexcept> // For std::invalid_argument

// Define constant from original CUDA code, now explicitly in C++ wrapper
#define BACKWARD_W_BATCH_THREADS 32

// Define a macro similar to CHECK_CUDA for Metal tensors
#define CHECK_MPS(x) TORCH_CHECK(x.is_mps(), #x " must be an MPS tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_MPS(x);    \
    CHECK_CONTIGUOUS(x)

// Helper function for ceiling division
template <typename T>
T ceil_div(const T x, const T y) {
    return x / y + !!(x % y);
}

// Forward declarations for conceptual Metal kernel launch functions.
// These functions are placeholders that would conceptually dispatch the Metal kernels.
// In a real PyTorch Metal backend, these would be calls to ATen's internal
// kernel dispatch utilities that handle setting up Metal buffers,
// compute command encoders, and dispatching threads.
// The arguments here are designed to mimic what the Metal kernel expects.

template <typename scalar_t>
void launch_logic_layer_metal_forward_kernel(
    const torch::Tensor& x,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    torch::Tensor& y,
    uint y_rows,
    uint y_cols,
    uint x_cols,
    uint w_cols
) {
    (void)x; (void)a; (void)b; (void)w; (void)y; // Suppress unused parameter warnings
    (void)y_rows; (void)y_cols; (void)x_cols; (void)w_cols;
    TORCH_WARN("Conceptual Metal kernel launch: logic_layer_metal_forward_kernel");
}

template <typename scalar_t>
void launch_logic_layer_metal_backward_w_kernel(
    const torch::Tensor& x,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& grad_y,
    torch::Tensor& grad_w_4, // Renamed for clarity as per original usage
    uint grad_y_rows,
    uint grad_y_cols,
    uint x_cols,
    uint grad_w_dim1, // BACKWARD_W_BATCH_THREADS
    uint grad_w_dim2  // 4
) {
    (void)x; (void)a; (void)b; (void)grad_y; (void)grad_w_4;
    (void)grad_y_rows; (void)grad_y_cols; (void)x_cols; (void)grad_w_dim1; (void)grad_w_dim2;
    TORCH_WARN("Conceptual Metal kernel launch: logic_layer_metal_backward_w_kernel");
}

template <typename scalar_t>
void launch_logic_layer_metal_backward_x_kernel(
    const torch::Tensor& x,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w,
    const torch::Tensor& grad_y,
    torch::Tensor& grad_x,
    const torch::Tensor& given_x_indices_of_y_start,
    const torch::Tensor& given_x_indices_of_y,
    uint grad_x_rows,
    uint grad_x_cols,
    uint w_cols
) {
    (void)x; (void)a; (void)b; (void)w; (void)grad_y; (void)grad_x;
    (void)given_x_indices_of_y_start; (void)given_x_indices_of_y;
    (void)grad_x_rows; (void)grad_x_cols; (void)w_cols;
    TORCH_WARN("Conceptual Metal kernel launch: logic_layer_metal_backward_x_kernel");
}

template <typename scalar_t>
void launch_logic_layer_metal_eval_kernel(
    const torch::Tensor& x,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& w_uint8, // Renamed to indicate uint8_t type
    torch::Tensor& y,
    uint y_rows,
    uint y_cols,
    uint x_cols
) {
    (void)x; (void)a; (void)b; (void)w_uint8; (void)y;
    (void)y_rows; (void)y_cols; (void)x_cols;
    TORCH_WARN("Conceptual Metal kernel launch: logic_layer_metal_eval_kernel");
}

template <typename scalar_t>
void launch_tensor_packbits_metal_kernel(
    const torch::Tensor& t_bool, // Renamed to indicate bool type
    torch::Tensor& b,
    uint t_rows,
    uint t_cols,
    uint b_cols,
    uint bit_count_val
) {
    (void)t_bool; (void)b;
    (void)t_rows; (void)t_cols; (void)b_cols; (void)bit_count_val;
    TORCH_WARN("Conceptual Metal kernel launch: tensor_packbits_metal_kernel");
}

template <typename scalar_t>
void launch_groupbitsum_metal_kernel(
    const torch::Tensor& b,
    torch::Tensor& t,
    uint t_rows,
    uint t_cols,
    uint b_rows,
    uint b_cols
) {
    (void)b; (void)t;
    (void)t_rows; (void)t_cols; (void)b_rows; (void)b_cols;
    TORCH_WARN("Conceptual Metal kernel launch: groupbitsum_metal_kernel");
}


// Public API for PyTorch extension: these functions are called from Python.

torch::Tensor logic_layer_metal_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);

    const auto batch_size = x.size(1);
    const auto out_size = w.size(0); // Number of neurons in the layer

    // Create the output tensor 'y' on the Metal device with the same dtype as 'x'.
    auto y = torch::empty({out_size, batch_size}, x.options());

    // Dispatch to the Metal kernel based on the scalar type of the input 'x'.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_metal_forward", ([&] {
        launch_logic_layer_metal_forward_kernel<scalar_t>(
            x, a, b, w, y,
            (uint)y.size(0), // y_rows (out_size)
            (uint)y.size(1), // y_cols (batch_size)
            (uint)x.size(1), // x_cols (batch_size) - used for x_data access
            (uint)w.size(1)  // w_cols (16)
        );
    }));

    // PyTorch's Metal backend typically handles error checking after kernel dispatch.
    return y;
}

torch::Tensor logic_layer_metal_backward_w(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);

    const auto batch_size = x.size(1);
    const auto out_size = grad_y.size(0);

    // Create a temporary tensor `grad_w_4` for intermediate gradient accumulation.
    // It will be (out_size, BACKWARD_W_BATCH_THREADS, 4)
    auto grad_w_4 = torch::empty({out_size, (long)BACKWARD_W_BATCH_THREADS, 4}, x.options());

    // Dispatch to the Metal kernel.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_metal_backward_w", ([&] {
        launch_logic_layer_metal_backward_w_kernel<scalar_t>(
            x, a, b, grad_y, grad_w_4,
            (uint)grad_y.size(0), // grad_y_rows (out_size)
            (uint)grad_y.size(1), // grad_y_cols (batch_size)
            (uint)x.size(1),      // x_cols (batch_size) - used for x_data access
            (uint)BACKWARD_W_BATCH_THREADS, // grad_w_dim1
            (uint)grad_w_4.size(2)  // grad_w_dim2 (4)
        );
    }));

    // Perform the sum and stacking operations on the host (PyTorch tensor operations).
    // These operations are already efficient on the PyTorch backend (including MPS if enabled).
    const auto grad_w_components = grad_w_4.sum(/*dim=*/1); // Sum along the BACKWARD_W_BATCH_THREADS dimension
    const auto grad_w_ab = grad_w_components.index({torch::indexing::Slice(), 0});
    const auto grad_w_a = grad_w_components.index({torch::indexing::Slice(), 1});
    const auto grad_w_b = grad_w_components.index({torch::indexing::Slice(), 2});
    const auto grad_w_ = grad_w_components.index({torch::indexing::Slice(), 3});

    // Reconstruct the final `grad_w` tensor based on the original logic.
    // torch::stack creates a new dimension, ensuring the output shape is correct (out_size, 16).
    torch::Tensor result_grad_w;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_metal_backward_w_stack", ([&] {
        result_grad_w = torch::stack({
            torch::zeros({out_size}, x.options()), // w_0 always 0 gradient
            grad_w_ab,
            grad_w_a - grad_w_ab,
            grad_w_a,
            grad_w_b - grad_w_ab,
            grad_w_b,
            grad_w_a + grad_w_b - static_cast<scalar_t>(2) * grad_w_ab,
            grad_w_a + grad_w_b - grad_w_ab,
            grad_w_ - grad_w_a - grad_w_b + grad_w_ab,
            grad_w_ - grad_w_a - grad_w_b + static_cast<scalar_t>(2) * grad_w_ab,
            grad_w_ - grad_w_b,
            grad_w_ - grad_w_b + grad_w_ab,
            grad_w_ - grad_w_a,
            grad_w_ - grad_w_a + grad_w_ab,
            grad_w_ - grad_w_ab,
            grad_w_,
        }, 1);
    }));
    return result_grad_w;
}

torch::Tensor logic_layer_metal_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(w);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);

    // Create an empty tensor `grad_x` with the same shape and options as `x`.
    auto grad_x = torch::empty_like(x);

    // Dispatch to the Metal kernel.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "logic_layer_metal_backward_x", ([&] {
        launch_logic_layer_metal_backward_x_kernel<scalar_t>(
            x, a, b, w, grad_y, grad_x, given_x_indices_of_y_start, given_x_indices_of_y,
            (uint)grad_x.size(0), // grad_x_rows (in_size)
            (uint)grad_x.size(1), // grad_x_cols (batch_size)
            (uint)w.size(1)       // w_cols (16)
        );
    }));

    return grad_x;
}

// -----------------------------------------------------------------------------
// logic_layer_metal_eval  (patched)
// -----------------------------------------------------------------------------
torch::Tensor logic_layer_metal_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor w)        // 'w' is uint8 op-indices
{
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(w.is_mps(),        "w must be an MPS tensor");
    TORCH_CHECK(w.is_contiguous(), "w must be contiguous");
    TORCH_CHECK(w.scalar_type() == torch::kU8,
                "w must be uint8 for eval mode");

    const auto out_size = w.size(0);          // #neurons
    auto y = torch::zeros({out_size, x.size(1)}, x.options());

    // ---------- portable dispatch ----------
    #if defined(AT_DISPATCH_ALL_TYPES_AND_HALF)        // Torch ≥ 2.2
        AT_DISPATCH_ALL_TYPES_AND_HALF(
            x.scalar_type(), "logic_layer_metal_eval_kernel", ([&] {
                launch_logic_layer_metal_eval_kernel<scalar_t>(
                    x, a, b, w, y,
                    (uint)y.size(0),
                    (uint)y.size(1),
                    (uint)x.size(1));
            }));
    #else                                              // Torch 1.13 – 2.1
        AT_DISPATCH_ALL_TYPES_AND(
            at::kHalf,
            x.scalar_type(), "logic_layer_metal_eval_kernel", ([&] {
                launch_logic_layer_metal_eval_kernel<scalar_t>(
                    x, a, b, w, y,
                    (uint)y.size(0),
                    (uint)y.size(1),
                    (uint)x.size(1));
            }));
    #endif
    return y;
}



// -----------------------------------------------------------------------------
// tensor_packbits_metal
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, int> tensor_packbits_metal(
    torch::Tensor t,               // boolean input
    const int      bit_count)      // 8 / 16 / 32 / 64
{
    CHECK_INPUT(t);
    TORCH_CHECK(t.scalar_type() == torch::kBool,
                "Input tensor 't' must be bool for packbits.");

    const auto batch_in_size  = t.size(1);
    const auto out_size       = t.size(0);
    const auto batch_out_size = ceil_div(batch_in_size,
                                         static_cast<int64_t>(bit_count));
    const auto pad_len        = (bit_count - batch_in_size % bit_count) % bit_count;

    auto dispatch_type = [bit_count]() {
        switch (bit_count) {
            case  8: return torch::kInt8;
            case 16: return torch::kInt16;
            case 32: return torch::kInt32;
            case 64: return torch::kInt64;
            default:
                throw std::invalid_argument("bit_count must be 8, 16, 32 or 64");
        }
    }();

    auto b = torch::zeros({out_size, batch_out_size},
                          t.options().dtype(dispatch_type));

    // ---------- portable dispatch ----------
    #if defined(AT_DISPATCH_INTEGRAL_TYPES)
        AT_DISPATCH_INTEGRAL_TYPES(
            b.scalar_type(), "tensor_packbits_metal_kernel", ([&] {
                launch_tensor_packbits_metal_kernel<scalar_t>(
                    t, b,
                    (uint)t.size(0),
                    (uint)t.size(1),
                    (uint)b.size(1),
                    (uint)bit_count);
            }));
    #else
        // Really ancient Torch (pre-1.4) fallback
        AT_DISPATCH_ALL_TYPES(
            b.scalar_type(), "tensor_packbits_metal_kernel", ([&] {
                launch_tensor_packbits_metal_kernel<scalar_t>(
                    t, b,
                    (uint)t.size(0),
                    (uint)t.size(1),
                    (uint)b.size(1),
                    (uint)bit_count);
            }));
    #endif
        return {b, static_cast<int>(pad_len)};
}

// -----------------------------------------------------------------------------
// groupbitsum_metal
// -----------------------------------------------------------------------------
torch::Tensor groupbitsum_metal(
    torch::Tensor b,   // packed tensor
    const int     pad_len,
    const int     k)   // number of groups
{
    CHECK_INPUT(b);
    TORCH_CHECK(at::isIntegralType(b.scalar_type(), /*includeBool=*/false),
                "'b' must be an integral tensor for groupbitsum.");

    const int  bit_count       = 8 * static_cast<int>(b.element_size());
    const auto batch_out_size  = b.size(1) * bit_count - pad_len;
    TORCH_CHECK(b.size(0) % k == 0,
                "b.size(0) must be divisible by k.");

    auto t = torch::zeros({k, batch_out_size},
                          b.options().dtype(torch::kInt32));

    #if defined(AT_DISPATCH_INTEGRAL_TYPES)
        AT_DISPATCH_INTEGRAL_TYPES(
            b.scalar_type(), "groupbitsum_metal_kernel", ([&] {
                launch_groupbitsum_metal_kernel<scalar_t>(
                    b, t,
                    (uint)t.size(0),
                    (uint)t.size(1),
                    (uint)b.size(0),
                    (uint)b.size(1));
            }));
    #else
        AT_DISPATCH_ALL_TYPES(
            b.scalar_type(), "groupbitsum_metal_kernel", ([&] {
                launch_groupbitsum_metal_kernel<scalar_t>(
                    b, t,
                    (uint)t.size(0),
                    (uint)t.size(1),
                    (uint)b.size(0),
                    (uint)b.size(1));
            }));
    #endif
    return t.transpose(0, 1).contiguous();
}



// PYBIND11_MODULE macro exposes the C++ functions to Python.
// TORCH_EXTENSION_NAME is a macro defined by PyTorch's build system.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &logic_layer_metal_forward, "Logic Layer forward (Metal)");
    m.def("backward_w", &logic_layer_metal_backward_w, "Logic Layer backward W (Metal)");
    m.def("backward_x", &logic_layer_metal_backward_x, "Logic Layer backward X (Metal)");
    m.def("eval", &logic_layer_metal_eval, "Logic Layer eval (Metal)");
    m.def("packbits", &tensor_packbits_metal, "Tensor Packbits (Metal)");
    m.def("groupbitsum", &groupbitsum_metal, "Group Bit Sum (Metal)");
}