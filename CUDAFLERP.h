/*******************************************************************
*   CUDAFLERP.h
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

#pragma once

#include "cuda_runtime.h"

#include <cstdint>

#ifdef __INTELLISENSE__
#include <algorithm>
#define asm(x)
#include "device_launch_parameters.h"
#define __CUDACC__
#include "device_functions.h"
#undef __CUDACC__
#endif

void CUDAFLERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, float* __restrict const d_out, const uint32_t neww, const uint32_t newh);

void CUDAFNERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, float* __restrict const d_out, const uint32_t neww, const uint32_t newh);
