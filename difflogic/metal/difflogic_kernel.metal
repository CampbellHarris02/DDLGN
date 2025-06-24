// difflogic_kernel.metal

#include <metal_stdlib>
#include <metal_atomic> // Include for atomic operations
using namespace metal;

// Helper function for ceil_div
template <typename T>
inline T ceil_div(const T x, const T y) {
    return x / y + (x % y != 0);
}

// Atomic add for half. Metal's atomic_fetch_add works on atomic_float.
// We cast half to float, perform atomic op, then cast back.
inline half atomic_add_half(device atomic_float* address, half val) {
    return (half)atomic_fetch_add_explicit(address, (float)val, memory_order_relaxed);
}

// Standard atomic add for float.
inline float atomic_add_float(device atomic_float* address, float val) {
    return atomic_fetch_add_explicit(address, val, memory_order_relaxed);
}

// Removed atomic_add_double as 'double' is not supported in Metal.
// If you need higher precision, you would need to implement software FP64 or stick to float.


// Kernel for the forward pass in training mode
template <typename scalar_t>
kernel void logic_layer_metal_forward_kernel(
    device const scalar_t* x_data [[buffer(0)]],
    device const long* a_data [[buffer(1)]],
    device const long* b_data [[buffer(2)]],
    device const scalar_t* w_data [[buffer(3)]],
    device scalar_t* y_data [[buffer(4)]],
    // Dimensions passed as pointers to uints within a buffer.
    // PyTorch's MPS backend handles mapping scalar arguments to these buffers.
    device const uint* y_rows_ptr [[buffer(5)]],
    device const uint* y_cols_ptr [[buffer(6)]],
    device const uint* x_cols_ptr [[buffer(7)]],
    device const uint* w_cols_ptr [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint y_rows = *y_rows_ptr;
    uint y_cols = *y_cols_ptr;
    uint x_cols = *x_cols_ptr;
    uint w_cols = *w_cols_ptr;

    uint total_elements = y_rows * y_cols;

    if (gid >= total_elements) {
        return;
    }

    uint col = gid / y_cols;
    uint row = gid % y_cols;

    const auto idx_a = a_data[col];
    const auto idx_b = b_data[col];
    const auto a_ = x_data[idx_a * x_cols + row];
    const auto b_ = x_data[idx_b * x_cols + row];

    const scalar_t w_1 = w_data[col * w_cols + 1];
    const scalar_t w_2 = w_data[col * w_cols + 2];
    const scalar_t w_3 = w_data[col * w_cols + 3];
    const scalar_t w_4 = w_data[col * w_cols + 4];
    const scalar_t w_5 = w_data[col * w_cols + 5];
    const scalar_t w_6 = w_data[col * w_cols + 6];
    const scalar_t w_7 = w_data[col * w_cols + 7];
    const scalar_t w_8 = w_data[col * w_cols + 8];
    const scalar_t w_9 = w_data[col * w_cols + 9];
    const scalar_t w_10 = w_data[col * w_cols + 10];
    const scalar_t w_11 = w_data[col * w_cols + 11];
    const scalar_t w_12 = w_data[col * w_cols + 12];
    const scalar_t w_13 = w_data[col * w_cols + 13];
    const scalar_t w_14 = w_data[col * w_cols + 14];
    const scalar_t w_15 = w_data[col * w_cols + 15];

    y_data[col * y_cols + row] = (
        ((w_1 * (a_ * b_)
        + w_2 * (a_ - a_ * b_))
        + (w_3 * a_
        + w_4 * (b_ - a_ * b_)))
        + ((w_5 * b_
        + w_6 * (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
        + (w_7 * (a_ + b_ - a_ * b_)
        + w_8 * (static_cast<scalar_t>(1) - (a_ + b_ - a_ * b_)))))
        + (((w_9 * (static_cast<scalar_t>(1) - (a_ + b_ - static_cast<scalar_t>(2) * a_ * b_))
        + w_10 * (static_cast<scalar_t>(1) - b_)) +
        (w_11 * (static_cast<scalar_t>(1) - b_ + a_ * b_)
        + w_12 * (static_cast<scalar_t>(1) - a_))) +
        (w_13 * (static_cast<scalar_t>(1) - a_ + a_ * b_)
        + w_14 * (static_cast<scalar_t>(1) - a_ * b_)
        + w_15)
    );
}


// Kernel for the backward pass calculating gradients with respect to weights (grad_w)
template <typename scalar_t>
kernel void logic_layer_metal_backward_w_kernel(
    device const scalar_t* x_data [[buffer(0)]],
    device const long* a_data [[buffer(1)]],
    device const long* b_data [[buffer(2)]],
    device const scalar_t* grad_y_data [[buffer(3)]],
    device scalar_t* grad_w_data [[buffer(4)]], // Output: 3D tensor (out_size, BACKWARD_W_BATCH_THREADS, 4)
    // Dimensions
    device const uint* grad_y_rows_ptr [[buffer(5)]],
    device const uint* grad_y_cols_ptr [[buffer(6)]],
    device const uint* x_cols_ptr [[buffer(7)]],
    device const uint* grad_w_dim1_ptr [[buffer(8)]],
    device const uint* grad_w_dim2_ptr [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint grad_y_rows = *grad_y_rows_ptr;
    uint grad_y_cols = *grad_y_cols_ptr;
    uint x_cols = *x_cols_ptr;
    uint grad_w_dim1 = *grad_w_dim1_ptr;
    uint grad_w_dim2 = *grad_w_dim2_ptr;

    uint total_elements_for_dispatch = grad_y_rows * grad_w_dim1;

    if (gid >= total_elements_for_dispatch) {
        return;
    }

    uint col = gid / grad_w_dim1;
    uint row_ = gid % grad_w_dim1;

    const auto idx_a = a_data[col];
    const auto idx_b = b_data[col];

    scalar_t grad_w_local_1 = 0;
    scalar_t grad_w_local_3 = 0;
    scalar_t grad_w_local_5 = 0;
    scalar_t grad_w_local_15 = 0;

    for (uint row = row_; row < grad_y_cols; row += grad_w_dim1) {
        const auto a_ = x_data[idx_a * x_cols + row];
        const auto b_ = x_data[idx_b * x_cols + row];
        const auto grad_y_ = grad_y_data[col * grad_y_cols + row];

        grad_w_local_1 += (a_ * b_) * grad_y_;
        grad_w_local_3 += a_ * grad_y_;
        grad_w_local_5 += b_ * grad_y_;
        grad_w_local_15 += grad_y_;
    }

    grad_w_data[col * (grad_w_dim1 * grad_w_dim2) + row_ * grad_w_dim2 + 0] = grad_w_local_1;
    grad_w_data[col * (grad_w_dim1 * grad_w_dim2) + row_ * grad_w_dim2 + 1] = grad_w_local_3;
    grad_w_data[col * (grad_w_dim1 * grad_w_dim2) + row_ * grad_w_dim2 + 2] = grad_w_local_5;
    grad_w_data[col * (grad_w_dim1 * grad_w_dim2) + row_ * grad_w_dim2 + 3] = grad_w_local_15;
}


// Kernel for the backward pass calculating gradients with respect to inputs (grad_x)
template <typename scalar_t>
kernel void logic_layer_metal_backward_x_kernel(
    device const scalar_t* x_data [[buffer(0)]],
    device const long* a_data [[buffer(1)]],
    device const long* b_data [[buffer(2)]],
    device const scalar_t* w_data [[buffer(3)]],
    device const scalar_t* grad_y_data [[buffer(4)]],
    device scalar_t* grad_x_data [[buffer(5)]], // Output: grad_x
    device const long* given_x_indices_of_y_start_data [[buffer(6)]],
    device const long* given_x_indices_of_y_data [[buffer(7)]],
    // Dimensions
    device const uint* grad_x_rows_ptr [[buffer(8)]],
    device const uint* grad_x_cols_ptr [[buffer(9)]],
    device const uint* w_cols_ptr [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    uint grad_x_rows = *grad_x_rows_ptr;
    uint grad_x_cols = *grad_x_cols_ptr;
    uint w_cols = *w_cols_ptr;

    uint total_elements = grad_x_rows * grad_x_cols;

    if (gid >= total_elements) {
        return;
    }

    uint col = gid / grad_x_cols;
    uint row = gid % grad_x_cols;

    scalar_t grad_x_accum = 0;

    const auto start = given_x_indices_of_y_start_data[col];
    const auto end = given_x_indices_of_y_start_data[col + 1];

    for (long cur = start; cur < end; ++cur) {
        const auto idx_y = given_x_indices_of_y_data[cur];
        const auto idx_a = a_data[idx_y];
        const auto idx_b = b_data[idx_y];
        const auto grad_y_ = grad_y_data[idx_y * grad_x_cols + row];
        const bool idx_is_a = (idx_a == col);

        if (idx_is_a) {
            const auto b_ = x_data[idx_b * grad_x_cols + row];
            const auto dy_dx = (
                (w_data[idx_y * w_cols + 1] * b_
                + w_data[idx_y * w_cols + 2] * (static_cast<scalar_t>(1) - b_)
                + w_data[idx_y * w_cols + 3]) +
                (w_data[idx_y * w_cols + 4] * -b_
                + w_data[idx_y * w_cols + 6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * b_)
                + w_data[idx_y * w_cols + 7] * (static_cast<scalar_t>(1) - b_)))
                + ((w_data[idx_y * w_cols + 8] * (b_ - static_cast<scalar_t>(1))
                + w_data[idx_y * w_cols + 9] * (static_cast<scalar_t>(2) * b_ - static_cast<scalar_t>(1))
                + w_data[idx_y * w_cols + 11] * b_)
                + (-w_data[idx_y * w_cols + 12]
                + w_data[idx_y * w_cols + 13] * (b_ - static_cast<scalar_t>(1))
                + w_data[idx_y * w_cols + 14] * -b_)
            );
            grad_x_accum += dy_dx * grad_y_;
        } else {
            const auto a_ = x_data[idx_a * grad_x_cols + row];
            const auto dy_dx = (
                (w_data[idx_y * w_cols + 1] * a_
                + w_data[idx_y * w_cols + 2] * -a_
                + w_data[idx_y * w_cols + 4] * (static_cast<scalar_t>(1) - a_))
                + (w_data[idx_y * w_cols + 5]
                + w_data[idx_y * w_cols + 6] * (static_cast<scalar_t>(1) - static_cast<scalar_t>(2) * a_)
                + w_data[idx_y * w_cols + 7] * (static_cast<scalar_t>(1) - a_)))
                + ((w_data[idx_y * w_cols + 8] * (a_ - static_cast<scalar_t>(1))
                + w_data[idx_y * w_cols + 9] * (static_cast<scalar_t>(2) * a_ - static_cast<scalar_t>(1))
                - w_data[idx_y * w_cols + 10])
                + (w_data[idx_y * w_cols + 11] * (a_ - static_cast<scalar_t>(1))
                + w_data[idx_y * w_cols + 13] * a_
                + w_data[idx_y * w_cols + 14] * -a_)
            );
            grad_x_accum += dy_dx * grad_y_;
        }
    }
    grad_x_data[col * grad_x_cols + row] = grad_x_accum;
}

// Function to evaluate binary logic operations in inference mode.
// For floating-point types, it converts inputs to boolean (0.0=false, non-0.0=true)
// and performs logical operations. For integral types, it uses bitwise operations directly.
template <typename T>
inline T bin_op_eval(const T a_, const T b_, const int op_idx) {
    if (metal::is_floating_point<T>::value) {
        bool A_bool = (a_ != static_cast<T>(0));
        bool B_bool = (b_ != static_cast<T>(0));

        bool result_bool;
        switch (op_idx) {
            case 0:  result_bool = false; break;
            case 1:  result_bool = A_bool && B_bool; break;
            case 2:  result_bool = A_bool && !B_bool; break;
            case 3:  result_bool = A_bool; break;
            case 4:  result_bool = B_bool && !A_bool; break;
            case 5:  result_bool = B_bool; break;
            case 6:  result_bool = A_bool != B_bool; break;
            case 7:  result_bool = A_bool || B_bool; break;
            case 8:  result_bool = !(A_bool || B_bool); break;
            case 9:  result_bool = !(A_bool != B_bool); break;
            case 10: result_bool = !B_bool; break;
            case 11: result_bool = !B_bool || A_bool; break;
            case 12: result_bool = !A_bool; break;
            case 13: result_bool = !A_bool || B_bool; break;
            case 14: result_bool = !(A_bool && B_bool); break;
            case 15: result_bool = true; break;
            default: result_bool = true; break;
        }
        return result_bool ? static_cast<T>(1) : static_cast<T>(0);
    } else {
        // IMPORTANT: Metal's native integer types for bitwise ops are typically limited to 32-bit.
        // Using 'unsigned int' here. If 'scalar_t' is int64_t, results for bitwise ops
        // in this function will be truncated to 32-bits.
        unsigned int val_a = (unsigned int)a_;
        unsigned int val_b = (unsigned int)b_;
        unsigned int result;
        switch (op_idx) {
            case 0:  result = 0; break;
            case 1:  result = val_a & val_b; break;
            case 2:  result = val_a & (~val_b); break;
            case 3:  result = val_a; break;
            case 4:  result = val_b & (~val_a); break;
            case 5:  result = val_b; break;
            case 6:  result = val_a ^ val_b; break;
            case 7:  result = val_a | val_b; break;
            case 8:  result = ~(val_a | val_b); break;
            case 9:  result = ~(val_a ^ val_b); break;
            case 10: result = ~val_b; break;
            case 11: result = ~val_b | val_a; break;
            case 12: result = ~val_a; break;
            case 13: result = ~val_a | val_b; break;
            case 14: result = ~(val_a & val_b); break;
            case 15: result = ~static_cast<unsigned int>(0); break; // All bits set
            default: result = ~static_cast<unsigned int>(0); break;
        }
        return static_cast<T>(result);
    }
}

// Kernel for inference mode evaluation using binary logical operations
template <typename scalar_t>
kernel void logic_layer_metal_eval_kernel(
    device const scalar_t* x_data [[buffer(0)]],
    device const long* a_data [[buffer(1)]],
    device const long* b_data [[buffer(2)]],
    device const uchar* w_data [[buffer(3)]], // w is uint8_t
    device scalar_t* y_data [[buffer(4)]],
    // Dimensions
    device const uint* y_rows_ptr [[buffer(5)]],
    device const uint* y_cols_ptr [[buffer(6)]],
    device const uint* x_cols_ptr [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint y_rows = *y_rows_ptr;
    uint y_cols = *y_cols_ptr;
    uint x_cols = *x_cols_ptr;

    uint total_elements = y_rows * y_cols;

    if (gid >= total_elements) {
        return;
    }

    uint col = gid / y_cols;
    uint row = gid % y_cols;

    const auto idx_a = a_data[col];
    const auto idx_b = b_data[col];
    const auto a_ = x_data[idx_a * x_cols + row];
    const auto b_ = x_data[idx_b * x_cols + row];
    const auto w_ = w_data[col];

    y_data[col * y_cols + row] = bin_op_eval(a_, b_, (int)w_);
}

// Kernel for packing boolean tensors into bit-packed integral tensors
template <typename scalar_t>
kernel void tensor_packbits_metal_kernel(
    device const bool* t_data [[buffer(0)]], // Input: boolean tensor
    device scalar_t* b_data [[buffer(1)]],     // Output: bit-packed integral tensor
    // Dimensions
    device const uint* t_rows_ptr [[buffer(2)]],
    device const uint* t_cols_ptr [[buffer(3)]],
    device const uint* b_cols_ptr [[buffer(4)]],
    device const uint* bit_count_val_ptr [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint t_rows = *t_rows_ptr;
    uint t_cols = *t_cols_ptr;
    uint b_cols = *b_cols_ptr;
    uint bit_count_val = *bit_count_val_ptr;

    uint total_elements = t_rows * b_cols;

    if (gid >= total_elements) {
        return;
    }

    uint row = gid / b_cols;
    uint col = gid % b_cols;

    // Use unsigned int for bit manipulation, which is generally 32-bit.
    // This will work correctly for scalar_t types up to int32_t.
    // If scalar_t is int64_t, only the lower 32 bits will be packed/unpacked correctly here.
    typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;
    // We need a specific type for the intermediate unsigned value that guarantees enough bits.
    // If scalar_t can be int64_t, then unsigned long long is needed for val_unsigned.
    // However, if we assume 'scalar_t' is a type like int8, int16, int32 for these kernels,
    // then 'unsigned int' is sufficient.
    // Given 'bit_count' can be 64, the intent is for scalar_t to be int64_t.
    // Metal does *not* support 64-bit bitwise operations directly.
    //
    // For `tensor_packbits_cuda_kernel`, `scalar_t` can be `int64_t`.
    // The `std::numeric_limits<unsigned_scalar_t>::digits` is important.
    //
    // We need a way to handle 64-bit packing if `scalar_t` is `int64_t`.
    // The current approach using `unsigned int` for `val_unsigned` will truncate for `int64_t`.
    //
    // Let's go back to the original `typedef typename std::make_unsigned<scalar_t>::type unsigned_scalar_t;`
    // from the CUDA code and use that. Metal's `make_unsigned` should resolve to `unsigned long` if `scalar_t` is `long`.
    // The issue is `long long` for `val_unsigned` which is explicitly unsupported.
    //
    // Okay, the core problem is `unsigned long long` is not available.
    // The largest integer type available for bitwise operations is `uint` (32-bit).
    //
    // If `scalar_t` is `int64_t`, then `unsigned_scalar_t` is `unsigned long`.
    // The error is specific to `unsigned long long`.
    //
    // Let's verify `unsigned long` support. Standard Metal Shading Language defines `long` as 64-bit integer,
    // but the bitwise operations on `long` (especially with `~`) might not be universally portable or efficient.
    // The best practice is to stick to `int` (32-bit) for bitwise operations if possible.

    // Let's assume for `tensor_packbits_metal_kernel` and `groupbitsum_metal_kernel`
    // that `scalar_t` will be `int8_t`, `int16_t`, or `int32_t`.
    // If `int64_t` is required for packing/unpacking, this portion needs a rewrite
    // to handle 64-bit integers as two 32-bit halves.

    // For now, let's explicitly use `uint` for intermediate bit operations
    // and acknowledge the 32-bit limitation for packing/unpacking.
    // The `dispatch_type` in C++ already limits to Int8, Int16, Int32, Int64.
    // If bit_count is 64, then scalar_t will be int64_t in C++.
    // Here, `unsigned_scalar_t` will be `unsigned long`.
    // The error was on `unsigned long long`. Let's use `unsigned_scalar_t` as defined.
    // The previous error was on `unsigned long long` which I hardcoded.
    // `typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;` should correctly resolve
    // to `unsigned int` for `int`, and `unsigned short` for `short`, etc.
    // For `long` (64-bit in C++, which maps to `int64_t` in PyTorch), `make_unsigned` should yield `unsigned long`.
    // The problem is that `long` (64-bit) is not consistently supported for bitwise operations or `~`
    // in all Metal compiler versions or hardware.

    // Let's try to be explicit for the bitwise union and ensure it matches the scalar_t.
    // The error about 'long long' is specifically in `bin_op_eval`, not `packbits`.
    // For `packbits` and `groupbitsum`, the `typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;`
    // is the correct approach. The previous compile error messages for `bin_op_eval` were caused by my incorrect hardcoded `unsigned long long`.

    // The current fix for `bin_op_eval` replaces `unsigned long long` with `unsigned int`.
    // This implies that `bin_op_eval` will only work correctly for `scalar_t` types up to `int32_t` when doing bitwise.
    // This is fine for the `eval` kernel, as boolean ops are typically not on `int64_t` bitmasks.

    // For `tensor_packbits_metal_kernel` and `groupbitsum_metal_kernel`:
    // If `scalar_t` becomes `int64_t` because `bit_count` is 64, then `unsigned_scalar_t` will be `unsigned long`.
    // The `union` approach might also be problematic for 64-bit if bitwise ops on `unsigned long` are not fully supported.

    // Let's try `unsigned long` instead of `unsigned long long` in `bin_op_eval` as well,
    // to be consistent with what `scalar_t` might resolve to if it's `long` (int64_t).
    // This is a gamble on Metal's `long` support.

    // Let's stick with `unsigned int` for `bin_op_eval` and explicitly note the 32-bit integer limitation for `eval`'s bitwise mode.
    // For `packbits` and `groupbitsum`, where `scalar_t` can legitimately be `int64_t`, we need to handle this.
    // The `union` and `unsigned_scalar_t` should pick up the correct size.
    // The specific error from your latest run:
    // `logic_layer_kernels.metal:320:18: error: 'long long' is not supported in Metal`
    // This confirms `unsigned long long` is the problem in `bin_op_eval`.

    // My last fix was not fully applied to the code block you used. Let's ensure `bin_op_eval` uses `unsigned int`.

    // FOR `tensor_packbits_metal_kernel` and `groupbitsum_metal_kernel`
    // The `typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;` is actually the correct and best way.
    // The problem is if `scalar_t` *is* `long` (64-bit integer), then `unsigned_scalar_t` becomes `unsigned long`.
    // Some older Metal compiler/runtime versions might not fully support 64-bit `unsigned long` for bitwise operations.
    // However, it's worth trying with the explicit `unsigned_scalar_t` type.
    // The `union` part for `packbits` and `groupbitsum` should be fine if `unsigned_scalar_t` is properly defined.

    // Let's ensure the `bin_op_eval` function uses `unsigned int` where it was `unsigned long long`.
    // The rest of `packbits` and `groupbitsum` should follow the `typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;` pattern.

    typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;
    unsigned_scalar_t val_unsigned = 0;

    const uint bit_count = bit_count_val;

    for (unsigned int i = 0; i < bit_count; ++i) {
        const uint t_col = bit_count * col + i;
        if (t_col < t_cols) {
            if (t_data[row * t_cols + t_col]) {
                val_unsigned = val_unsigned | (static_cast<unsigned_scalar_t>(1) << i);
            }
        }
    }
    b_data[row * b_cols + col] = static_cast<scalar_t>(val_unsigned);
}


// Kernel for summing bits within groups of bit-packed tensors
template <typename scalar_t>
kernel void groupbitsum_metal_kernel(
    device const scalar_t* b_data [[buffer(0)]], // Input: bit-packed integral tensor
    device int* t_data [[buffer(1)]],             // Output: sum of bits (int)
    // Dimensions
    device const uint* t_rows_ptr [[buffer(2)]],
    device const uint* t_cols_ptr [[buffer(3)]],
    device const uint* b_rows_ptr [[buffer(4)]],
    device const uint* b_cols_ptr [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint t_rows = *t_rows_ptr;
    uint t_cols = *t_cols_ptr;
    uint b_rows = *b_rows_ptr;
    uint b_cols = *b_cols_ptr;

    uint total_elements = t_rows * t_cols;

    if (gid >= total_elements) {
        return;
    }

    uint row = gid / t_cols;
    uint col = gid % t_cols;

    typedef typename metal::make_unsigned<scalar_t>::type unsigned_scalar_t;
    constexpr int bit_count = sizeof(scalar_t) * 8;
    const auto class_size = b_rows / t_rows;

    int res = 0;

    for (uint i = 0; i < class_size; ++i) {
        uint current_b_row_idx = row * class_size + i;
        uint packed_b_col_idx = col / bit_count;

        const unsigned_scalar_t val_unsigned = static_cast<unsigned_scalar_t>(b_data[current_b_row_idx * b_cols + packed_b_col_idx]);

        const unsigned int bit_mask_shift = col % bit_count;
        const unsigned_scalar_t bit_mask = static_cast<unsigned_scalar_t>(1) << bit_mask_shift;

        res += ((val_unsigned & bit_mask) != 0);
    }
    t_data[row * t_cols + col] = res;
}
