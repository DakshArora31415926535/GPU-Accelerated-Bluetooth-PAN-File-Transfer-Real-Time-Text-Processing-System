/**
 * gpu_text_pipeline.cu
 *
 * GPU-accelerated text normalization, transformation, and analysis
 * used in the Bluetooth PAN File Transfer & Automated Typing System.
 *
 * The idea: once a text payload is received over TCP, we offload the
 * expensive normalization / statistics phase to the GPU to keep the
 * CPU free for networking and input injection.
 *
 * This file intentionally contains multiple CUDA kernels, grid-stride
 * loops, shared-memory reductions, and atomic operations in order to
 * demonstrate realistic GPU-based text processing.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Utility: CUDA error checking
// -----------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA] Error %d (%s) at %s:%d in %s\n",           \
                    static_cast<int>(_err), cudaGetErrorString(_err),          \
                    __FILE__, __LINE__, #expr);                                \
            throw std::runtime_error("CUDA failure");                          \
        }                                                                      \
    } while (0)

// -----------------------------------------------------------------------------
// Device constants
// -----------------------------------------------------------------------------

// We limit histogram to first 128 ASCII codes for analysis.
constexpr int ASCII_HISTOGRAM_SIZE = 128;

// -----------------------------------------------------------------------------
// Kernel 1: Normalize + uppercase
//
// - Converts all lowercase a–z to A–Z
// - Maps control characters and non-printables to ' ' (space)
// - Leaves basic punctuation and digits untouched
// -----------------------------------------------------------------------------

__global__ void normalize_and_uppercase_kernel(
    const char* __restrict__ in,
    char* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char c = static_cast<unsigned char>(in[idx]);

    // Map non-printable ASCII to space
    if (c < 32 || c == 127) {
        c = ' ';
    }

    // Convert tabs/newlines to space as well
    if (c == '\t' || c == '\r' || c == '\n') {
        c = ' ';
    }

    // ASCII lowercase -> uppercase
    if (c >= 'a' && c <= 'z') {
        c = static_cast<unsigned char>(c - 32);
    }

    out[idx] = static_cast<char>(c);
}

// -----------------------------------------------------------------------------
// Kernel 2: ASCII histogram (0..127)
//
// Uses shared memory per-block to reduce contention on global memory, then
// atomically accumulates into global_hist.
// -----------------------------------------------------------------------------

__global__ void ascii_histogram_kernel(
    const char* __restrict__ data,
    int n,
    unsigned int* __restrict__ global_hist)
{
    extern __shared__ unsigned int local_hist[];

    // Initialize shared memory histogram
    for (int i = threadIdx.x; i < ASCII_HISTOGRAM_SIZE; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    // Grid-stride loop over input text
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (global_idx < n) {
        unsigned char c = static_cast<unsigned char>(data[global_idx]);

        if (c < ASCII_HISTOGRAM_SIZE) {
            atomicAdd(&local_hist[c], 1u);
        }

        global_idx += stride;
    }
    __syncthreads();

    // Accumulate shared histogram into global histogram
    for (int i = threadIdx.x; i < ASCII_HISTOGRAM_SIZE; i += blockDim.x) {
        unsigned int val = local_hist[i];
        if (val > 0) {
            atomicAdd(&global_hist[i], val);
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 3: Rolling hash (very simple XOR-based hash)
//
// Not cryptographically secure – just a fast content fingerprint.
//
// Each thread computes a per-character contribution and atomically XORs it
// into a single 64-bit value.
// -----------------------------------------------------------------------------

__global__ void rolling_hash_kernel(
    const char* __restrict__ data,
    int n,
    unsigned long long* __restrict__ hash_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned char c = static_cast<unsigned char>(data[idx]);
    // Simple mixing function
    unsigned long long contribution =
        (static_cast<unsigned long long>(c) + 1ull) * 1315423911ull;

    atomicXor(hash_out, contribution);
}

// -----------------------------------------------------------------------------
// Host-side helper: allocate pinned host memory for better transfer speed
// -----------------------------------------------------------------------------

static char* allocate_pinned_buffer(int n)
{
    char* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, static_cast<size_t>(n)));
    return ptr;
}

// -----------------------------------------------------------------------------
// Public API (C-compatible):
//
// gpu_process_text
//
// - input: raw text buffer (`input`, `length`)
// - output: newly allocated text buffer (`*output`, `*out_length`)
// - rolling_hash_out: 64-bit hash of normalized text
// - total_time_ms: measured GPU time (kernels only)
//
// NOTE:
//   - `*output` is allocated using cudaMallocHost (pinned memory). Caller
//     should free it using cudaFreeHost() or a small wrapper.
// -----------------------------------------------------------------------------

extern "C" void gpu_process_text(
    const char* input,
    int length,
    char** output,
    int* out_length,
    unsigned long long* rolling_hash_out,
    float* total_time_ms)
{
    if (!input || length <= 0 || !output || !out_length ||
        !rolling_hash_out || !total_time_ms) {
        throw std::invalid_argument("gpu_process_text: invalid argument");
    }

    // -------------------------------------------------------------------------
    // Allocate host output buffer (pinned) and device buffers
    // -------------------------------------------------------------------------

    char* h_out = allocate_pinned_buffer(length);
    *output = h_out;
    *out_length = length;  // We keep length unchanged in this pipeline

    char* d_in = nullptr;
    char* d_out = nullptr;
    unsigned int* d_hist = nullptr;
    unsigned long long* d_hash = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, static_cast<size_t>(length)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(length)));
    CUDA_CHECK(cudaMalloc(&d_hist, sizeof(unsigned int) * ASCII_HISTOGRAM_SIZE));
    CUDA_CHECK(cudaMalloc(&d_hash, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_hist, 0, sizeof(unsigned int) * ASCII_HISTOGRAM_SIZE));
    CUDA_CHECK(cudaMemset(d_hash, 0, sizeof(unsigned long long)));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, input, static_cast<size_t>(length),
                          cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Configure grid and block sizes
    // -------------------------------------------------------------------------

    const int block_size = 256;
    const int grid_size = (length + block_size - 1) / block_size;

    // -------------------------------------------------------------------------
    // CUDA events for timing
    // -------------------------------------------------------------------------

    cudaEvent_t start_evt, stop_evt;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt, 0));

    // -------------------------------------------------------------------------
    // Stage 1: normalize + uppercase
    // -------------------------------------------------------------------------

    normalize_and_uppercase_kernel<<<grid_size, block_size>>>(d_in, d_out, length);
    CUDA_CHECK(cudaGetLastError());

    // -------------------------------------------------------------------------
    // Stage 2: ASCII histogram (shared memory)
    // -------------------------------------------------------------------------

    size_t shared_bytes = sizeof(unsigned int) * ASCII_HISTOGRAM_SIZE;
    ascii_histogram_kernel<<<grid_size, block_size, shared_bytes>>>(
        d_out, length, d_hist);
    CUDA_CHECK(cudaGetLastError());

    // -------------------------------------------------------------------------
    // Stage 3: rolling hash
    // -------------------------------------------------------------------------

    rolling_hash_kernel<<<grid_size, block_size>>>(d_out, length, d_hash);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_evt, 0));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt));

    // -------------------------------------------------------------------------
    // Copy results back to host
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(h_out, d_out, static_cast<size_t>(length),
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned int> h_hist(ASCII_HISTOGRAM_SIZE);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist,
                          sizeof(unsigned int) * ASCII_HISTOGRAM_SIZE,
                          cudaMemcpyDeviceToHost));

    unsigned long long h_hash = 0;
    CUDA_CHECK(cudaMemcpy(&h_hash, d_hash,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // Optionally print a compact summary to stderr for logging/debugging.
    // (You can remove this if it is too verbose.)
    // -------------------------------------------------------------------------

    fprintf(stderr, "[GPU] Processed %d bytes in %.3f ms (kernels only)\n",
            length, elapsed_ms);
    fprintf(stderr, "[GPU] Rolling hash: 0x%016llX\n",
            static_cast<unsigned long long>(h_hash));

    // Example: print distribution of alphabetic characters (A–Z) and digits
    unsigned int alpha_count = 0;
    unsigned int digit_count = 0;

    for (int c = 'A'; c <= 'Z'; ++c) {
        alpha_count += h_hist[c];
    }
    for (int c = '0'; c <= '9'; ++c) {
        digit_count += h_hist[c];
    }

    fprintf(stderr, "[GPU] Alpha chars: %u, Digit chars: %u\n",
            alpha_count, digit_count);

    // -------------------------------------------------------------------------
    // Output values back to caller
    // -------------------------------------------------------------------------

    *rolling_hash_out = h_hash;
    *total_time_ms = elapsed_ms;

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start_evt));
    CUDA_CHECK(cudaEventDestroy(stop_evt));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_hash));
}

/**
 * Optional helper: free pinned buffer returned by gpu_process_text.
 * You can call this from C++ if you want a symmetric API.
 */
extern "C" void gpu_free_buffer(char* ptr)
{
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

#ifdef GPU_TEXT_PIPELINE_STANDALONE_TEST
// -----------------------------------------------------------------------------
// Minimal standalone test harness (optional).
// Compile this file with -DGPU_TEXT_PIPELINE_STANDALONE_TEST to run it
// as a self-contained example.
// -----------------------------------------------------------------------------
#include <iostream>

int main()
{
    std::string input = "Hello, world!\nthis is\ta test 1234.\nline 2.";

    char* out_buf = nullptr;
    int out_len = 0;
    unsigned long long hash = 0;
    float time_ms = 0.0f;

    try {
        gpu_process_text(input.c_str(),
                         static_cast<int>(input.size()),
                         &out_buf,
                         &out_len,
                         &hash,
                         &time_ms);
    } catch (const std::exception& ex) {
        std::cerr << "gpu_process_text failed: " << ex.what() << std::endl;
        return 1;
    }

    std::string normalized(out_buf, out_buf + out_len);
    std::cout << "Normalized text: [" << normalized << "]\n";
    std::cout << "Rolling hash: 0x" << std::hex << hash << std::dec << "\n";
    std::cout << "GPU time: " << time_ms << " ms\n";

    gpu_free_buffer(out_buf);
    return 0;
}
#endif
