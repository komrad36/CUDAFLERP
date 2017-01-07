/*******************************************************************
*   main.cpp
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

#include <chrono>
#include <cstring>
#include <iostream>

#include "CUDAFLERP.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

using namespace std::chrono;

#include <opencv2/opencv.hpp>

int main() {
	constexpr auto warmups = 2000;
	constexpr auto runs = 2000;

	auto image = new float[4];
	image[0] = 255.0f;
	image[1] = 255.0f;
	image[2] = 0.0f;
	image[3] = 0.0f;

	constexpr int oldw = 2;
	constexpr int oldh = 2;
	constexpr int neww = static_cast<int>(static_cast<double>(oldw) * 40.0);
	constexpr int newh = static_cast<int>(static_cast<double>(oldh) * 100.0);
	const size_t total = static_cast<size_t>(neww)*static_cast<size_t>(newh);

	// ------------- CUDAFLERP ------------

	// setting cache and shared modes
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	// allocating and transferring image and binding to texture object
	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* d_img_arr;
	cudaMallocArray(&d_img_arr, &chandesc_img, oldw, oldh);
	cudaMemcpyToArray(d_img_arr, 0, 0, image, sizeof(float) * static_cast<size_t>(oldw) * static_cast<size_t>(oldh), cudaMemcpyHostToDevice);
	struct cudaResourceDesc resdesc_img;
	memset(&resdesc_img, 0, sizeof(resdesc_img));
	resdesc_img.resType = cudaResourceTypeArray;
	resdesc_img.res.array.array = d_img_arr;
	struct cudaTextureDesc texdesc_img;
	memset(&texdesc_img, 0, sizeof(texdesc_img));
	texdesc_img.addressMode[0] = cudaAddressModeClamp;
	texdesc_img.addressMode[1] = cudaAddressModeClamp;
	texdesc_img.readMode = cudaReadModeElementType;
	texdesc_img.filterMode = cudaFilterModePoint;
	texdesc_img.normalizedCoords = 0;
	cudaTextureObject_t d_img_tex = 0;
	cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr);

	float* d_out = nullptr;
	cudaMalloc(&d_out, sizeof(float) * total);

	for (int i = 0; i < warmups; ++i) CUDAFLERP(d_img_tex, oldw, oldh, d_out, neww, newh);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) CUDAFLERP(d_img_tex, oldw, oldh, d_out, neww, newh);
	auto end = high_resolution_clock::now();
	auto sum = (end - start) / runs;

	auto h_out = new float[neww * newh];
	cudaMemcpy(h_out, d_out, sizeof(float)*total, cudaMemcpyDeviceToHost);

	std::cout << "CUDA reports " << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "CUDAFLERP took " << static_cast<double>(sum.count()) * 1e-3 << " us." << std::endl;

	std::cout << "Input stats: " << oldh << " rows, " << oldw << " cols." << std::endl;
	std::cout << "Output stats: " << newh << " rows, " << neww << " cols." << std::endl;
}
