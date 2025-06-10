#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

kernel void element_wise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] + b[index];
}

kernel void element_wise_multiply(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = a[index] * b[index];
}

kernel void scalar_multiply(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    result[index] = input[index] * scalar;
}

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 threads_per_grid [[threads_per_grid]]
) {
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    
    // Check bounds
    if (row >= M || col >= N) return;
    
    // Accumulate dot product
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += a[row * K + k] * b[k * N + col];
    }
    
    c[row * N + col] = sum;
}

// Tiled version for better performance
kernel void matmul_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 threads_per_threadgroup [[threads_per_threadgroup]],
    uint2 threadgroup_position [[threadgroup_position_in_grid]]
) {
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];
    
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    const uint local_row = thread_position.y % TILE_SIZE;
    const uint local_col = thread_position.x % TILE_SIZE;
    
    float sum = 0.0;
    
    // Iterate over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles cooperatively
        if (row < M && (t * TILE_SIZE + local_col) < K) {
            a_tile[local_row][local_col] = a[row * K + t * TILE_SIZE + local_col];
        } else {
            a_tile[local_row][local_col] = 0.0;
        }
        
        if (col < N && (t * TILE_SIZE + local_row) < K) {
            b_tile[local_row][local_col] = b[(t * TILE_SIZE + local_row) * N + col];
        } else {
            b_tile[local_row][local_col] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[local_row][k] * b_tile[k][local_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

kernel void matmul_batched(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint3 thread_position [[thread_position_in_grid]]
) {
    const uint batch_idx = thread_position.z;
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    
    if (batch_idx >= batch_size || row >= M || col >= N) return;
    
    const uint batch_offset = batch_idx * M * K;  // Offset for matrix A
    const uint b_batch_offset = batch_idx * K * N;  // Offset for matrix B
    const uint c_batch_offset = batch_idx * M * N;  // Offset for output matrix
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += a[batch_offset + row * K + k] * b[b_batch_offset + k * N + col];
    }
    
    c[c_batch_offset + row * N + col] = sum;
}

kernel void matmul_batched_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint3 thread_position [[thread_position_in_grid]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]]
) {
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];
    
    const uint batch_idx = thread_position.z;
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    const uint local_row = row % TILE_SIZE;
    const uint local_col = col % TILE_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    const uint batch_offset = batch_idx * M * K;
    const uint b_batch_offset = batch_idx * K * N;
    const uint c_batch_offset = batch_idx * M * N;
    
    float sum = 0.0;
    
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < M && (t * TILE_SIZE + local_col) < K) {
            a_tile[local_row][local_col] = a[batch_offset + row * K + t * TILE_SIZE + local_col];
        } else {
            a_tile[local_row][local_col] = 0.0;
        }
        
        if (col < N && (t * TILE_SIZE + local_row) < K) {
            b_tile[local_row][local_col] = b[b_batch_offset + (t * TILE_SIZE + local_row) * N + col];
        } else {
            b_tile[local_row][local_col] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[local_row][k] * b_tile[k][local_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        c[c_batch_offset + row * N + col] = sum;
    }
}

kernel void matmul_transposed(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant bool& transpose_a [[buffer(6)]],
    constant bool& transpose_b [[buffer(7)]],
    uint2 thread_position [[thread_position_in_grid]])
{
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        // If transposed, swap indices when accessing elements
        const uint a_idx = transpose_a ? (k * M + row) : (row * K + k);
        const uint b_idx = transpose_b ? (col * K + k) : (k * N + col);
        sum += a[a_idx] * b[b_idx];
    }
    
    c[row * N + col] = sum;
}

kernel void matmul_transposed_tiled(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant bool& transpose_a [[buffer(6)]],
    constant bool& transpose_b [[buffer(7)]],
    uint2 thread_position [[thread_position_in_grid]],
    uint2 threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup float a_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float b_tile[TILE_SIZE][TILE_SIZE];
    
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    const uint local_row = thread_position.y % TILE_SIZE;
    const uint local_col = thread_position.x % TILE_SIZE;
    
    float sum = 0.0;
    
    // Iterate over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles cooperatively, handling transposition
        if (row < M && (t * TILE_SIZE + local_col) < K) {
            uint a_idx = transpose_a ? 
                ((t * TILE_SIZE + local_col) * M + row) : 
                (row * K + t * TILE_SIZE + local_col);
            a_tile[local_row][local_col] = a[a_idx];
        } else {
            a_tile[local_row][local_col] = 0.0;
        }
        
        if (col < N && (t * TILE_SIZE + local_row) < K) {
            uint b_idx = transpose_b ? 
                (col * K + t * TILE_SIZE + local_row) : 
                ((t * TILE_SIZE + local_row) * N + col);
            b_tile[local_row][local_col] = b[b_idx];
        } else {
            b_tile[local_row][local_col] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[local_row][k] * b_tile[k][local_col];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

// Similarly update the batched versions
kernel void matmul_transposed_batched(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    constant bool& transpose_a [[buffer(7)]],
    constant bool& transpose_b [[buffer(8)]],
    uint3 thread_position [[thread_position_in_grid]])
{
    const uint batch_idx = thread_position.z;
    const uint row = thread_position.y;
    const uint col = thread_position.x;
    
    if (batch_idx >= batch_size || row >= M || col >= N) return;
    
    const uint batch_offset_a = batch_idx * M * K;
    const uint batch_offset_b = batch_idx * K * N;
    const uint batch_offset_c = batch_idx * M * N;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        const uint a_idx = batch_offset_a + (transpose_a ? (k * M + row) : (row * K + k));
        const uint b_idx = batch_offset_b + (transpose_b ? (col * K + k) : (k * N + col));
        sum += a[a_idx] * b[b_idx];
    }
    
    c[batch_offset_c + row * N + col] = sum;
}

// Convolution 2D kernel
kernel void conv2d(
    device const float* input_tensor [[buffer(0)]],    // Input tensor: [batch, in_channels, height, width]
    device const float* kernel_weights [[buffer(1)]],   // Kernel weights: [out_channels, in_channels, kernel_h, kernel_w]
    device float* output_tensor [[buffer(2)]],         // Output tensor: [batch, out_channels, out_height, out_width]
    constant uint& batch_size [[buffer(3)]],
    constant uint& in_channels [[buffer(4)]],
    constant uint& in_height [[buffer(5)]],
    constant uint& in_width [[buffer(6)]],
    constant uint& out_channels [[buffer(7)]],
    constant uint& kernel_height [[buffer(8)]],
    constant uint& kernel_width [[buffer(9)]],
    constant uint& out_height [[buffer(10)]],
    constant uint& out_width [[buffer(11)]],
    constant uint& stride_h [[buffer(12)]],
    constant uint& stride_w [[buffer(13)]],
    constant uint& pad_h [[buffer(14)]],
    constant uint& pad_w [[buffer(15)]],
    uint3 thread_position [[thread_position_in_grid]]
) {
    const uint batch = thread_position.z;
    const uint out_ch = thread_position.y;
    const uint out_x = thread_position.x / out_width;
    const uint out_y = thread_position.x % out_width;
    
    // Check bounds
    if (batch >= batch_size || out_ch >= out_channels || out_x >= out_height || out_y >= out_width) {
        return;
    }
    
    float sum = 0.0;
    
    // Perform convolution
    for (uint in_ch = 0; in_ch < in_channels; ++in_ch) {
        for (uint k_h = 0; k_h < kernel_height; ++k_h) {
            for (uint k_w = 0; k_w < kernel_width; ++k_w) {
                // Calculate input coordinates
                int in_x = int(out_x * stride_h) - int(pad_h) + int(k_h);
                int in_y = int(out_y * stride_w) - int(pad_w) + int(k_w);
                
                // Check if input coordinates are valid
                if (in_x >= 0 && in_x < int(in_height) && in_y >= 0 && in_y < int(in_width)) {
                    // Calculate indices
                    uint input_idx = batch * (in_channels * in_height * in_width) +
                                   in_ch * (in_height * in_width) +
                                   in_x * in_width + in_y;
                    
                    uint kernel_idx = out_ch * (in_channels * kernel_height * kernel_width) +
                                    in_ch * (kernel_height * kernel_width) +
                                    k_h * kernel_width + k_w;
                    
                    sum += input_tensor[input_idx] * kernel_weights[kernel_idx];
                }
            }
        }
    }
    
    // Write output
    uint output_idx = batch * (out_channels * out_height * out_width) +
                     out_ch * (out_height * out_width) +
                     out_x * out_width + out_y;
    output_tensor[output_idx] = sum;
} 