// rmsnorm_kernel.cu
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

template<typename scalar_t>
__device__ inline float to_float(scalar_t v) { return static_cast<float>(v); }
template<>
__device__ inline float to_float<__half>(__half v) { return __half2float(v); }

template<typename scalar_t>
__device__ inline scalar_t from_float(float v) { return static_cast<scalar_t>(v); }
template<>
__device__ inline __half from_float<__half>(float v) { return __float2half(v); }

template<typename scalar_t>
__global__ void rmsnorm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ y,
    int rows, int hidden, float eps
){
    extern __shared__ float s[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    const int base = row * hidden;
    float sumsq = 0.f;

    // reduce sum of squares across the hidden dimension
    for (int j = tid; j < hidden; j += blockDim.x) {
        float v = to_float<scalar_t>(x[base + j]);
        sumsq += v * v;
    }
    s[tid] = sumsq;
    __syncthreads();

    // parallel reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(s[0] / (float)hidden + eps);

    // write output
    for (int j = tid; j < hidden; j += blockDim.x) {
        float v  = to_float<scalar_t>(x[base + j]);
        float ww = to_float<scalar_t>(w[j]);
        y[base + j] = from_float<scalar_t>(v * inv_rms * ww);
    }
}

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "dtype mismatch");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.size(-1) == weight.numel(), "hidden_size mismatch");

    auto out = torch::empty_like(x);

    const int hidden = static_cast<int>(x.size(-1));
    const int64_t rows64 = x.numel() / hidden;
    TORCH_CHECK(rows64 <= INT_MAX, "too many rows");
    const int rows = static_cast<int>(rows64);

    int block = 256;
    if (hidden < block) {
        int b = 1;
        while (b * 2 <= hidden && b * 2 <= 256) b *= 2;
        block = b;
    }
    const dim3 grid(rows);
    size_t shmem = block * sizeof(float);

    at::cuda::CUDAGuard device_guard(x.get_device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::kHalf) {
        rmsnorm_forward_kernel<__half><<<grid, block, shmem, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr()),
            reinterpret_cast<const __half*>(weight.data_ptr()),
            reinterpret_cast<__half*>(out.data_ptr()),
            rows, hidden, static_cast<float>(eps)
        );
    } else if (x.scalar_type() == at::kFloat) {
        rmsnorm_forward_kernel<float><<<grid, block, shmem, stream>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            out.data_ptr<float>(),
            rows, hidden, static_cast<float>(eps)
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype (supported: float16, float32)");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

