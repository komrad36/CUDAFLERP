/*******************************************************************
*   CUDAFLERP.cu
*   CUDAFLERP
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Jan 7, 2017
*******************************************************************/
//
// The file CUDAFLERP.h exposes two extremely high performance GPU
// resize operations,
// CUDAFLERP (bilinear interpolation), and 
// CUDAFNERP (nearest neighbor interpolation), for 32-bit float
// grayscale data.
//
// For 8-bit unsigned integer data, see the CUDALERP project instead.
//
// CUDAFLERP offers superior accuracy to CUDA's built-in texture
// interpolator at comparable performance. The accuracy if compiled
// with -use-fast-math off is nearly equivalent to my CPU interpolator,
// KLERP, while still being as fast as the built-in interpolation.
// 
// Particularly for large images, CUDAFLERP dramatically outperforms
// even the highly tuned CPU AVX2 versions.
// 
// All functionality is contained in the header 'CUDAFLERP.h' and
// the source file 'CUDAFLERP.cu' and has no external dependencies at all.
// 
// The file 'main.cpp' is an example and speed test driver.
//

#include "CUDAFLERP.h"

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAFNERP_kernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, float* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = y*gys;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = x*gxs;
		const float res = tex2D<float>(d_img_tex, fx, fy);
		if (x < neww) d_out[y*neww + x] = res;
	}
}

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAFLERP_kernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, float* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = (y + 0.5f)*gys - 0.5f;
	const float wt_y = fy - floor(fy);
	const float invwt_y = 1.0f - wt_y;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = (x + 0.5f)*gxs - 0.5f;
		const float4 f = tex2Dgather<float4>(d_img_tex, fx + 0.5f, fy + 0.5f);
		const float wt_x = fx - floor(fx);
		const float invwt_x = 1.0f - wt_x;
		const float xa = invwt_x*f.w + wt_x*f.z;
		const float xb = invwt_x*f.x + wt_x*f.y;
		const float res = invwt_y*xa + wt_y*xb;
		if (x < neww) d_out[y*neww + x] = res;
	}
}

void CUDAFNERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, float* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	CUDAFNERP_kernel<<<{((neww - 1) >> 9) + 1, newh}, 256>>>(d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}

void CUDAFLERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, float* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	CUDAFLERP_kernel<<<{((neww - 1) >> 9) + 1, newh}, 256>>>(d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}
