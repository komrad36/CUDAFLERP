The file CUDAFLERP.h exposes two extremely high performance GPU
resize operations,
CUDAFLERP (bilinear interpolation), and 
CUDAFNERP (nearest neighbor interpolation), for 32-bit float
grayscale data.

For 8-bit unsigned integer data, see the CUDALERP project instead.

CUDAFLERP offers superior accuracy to CUDA's built-in texture
interpolator at comparable performance. The accuracy if compiled
with -use-fast-math off is nearly equivalent to my CPU interpolator,
KLERP, while still being as fast as the built-in interpolation.

Particularly for large images, CUDAFLERP dramatically outperforms
even the highly tuned CPU AVX2 versions.

All functionality is contained in the header 'CUDAFLERP.h' and
the source file 'CUDAFLERP.cu' and has no external dependencies at all.

The file 'main.cpp' is an example and speed test driver.
