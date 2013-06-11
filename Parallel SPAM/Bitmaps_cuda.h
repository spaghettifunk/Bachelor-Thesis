#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "Bitmap4.cuh"
#include "TreeNode.h"

// reduce
template <unsigned int blockSize>
__device__ void reduce(int *g_idata, 
					   int *g_odata, 
					   unsigned int n);

// header for AND each int bitwise
__global__ void AndBitwiseOperation(unsigned  int* _memory_device,  
									const int b1_size, 
									unsigned int* b1_memory, 
									unsigned int* b2_memory);

// BITMAP4
// header for Create s-bitmap4
__global__ void CreateSBitmap4(unsigned short* ms_device, 
							   unsigned short* msib_device,
							   int* _memSizeShort_device);

// headers for counting support of bitmap4
__global__ void Count4_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device);

__device__ int Count4_device(unsigned short* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device);

// BITMAP8
// header for Create s-bitmap8
__global__ void CreateSBitmap8(unsigned short* ms_device, 
							   unsigned short* msib_device,
							   int* _memSizeShort_device);

// headers for counting support of bitmap8
__global__ void Count8_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device);

__device__ int Count8_device(unsigned short* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device);

// BITMAP16
// header for Create s-bitmap16
__global__ void CreateSBitmap16(unsigned short* ms_device, 
							   unsigned short* msib_device,
							   int* _memSizeShort_device);

// headers for counting support of bitmap16
__global__ void Count16_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device);

__device__ int Count16_device(unsigned short* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device);


// BITMAP32
// header for Create s-bitmap32
__global__ void CreateSBitmap32(unsigned short* ms_device, 
							   unsigned short* msib_device,
							   int* _memSizeShort_device);

// headers for counting support of bitmap32
__global__ void Count32_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device);

__device__ int Count32_device(unsigned short* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device);

// BITMAP64
// header for Create s-bitmap64
__global__ void CreateSBitmap64(unsigned short* ms_device, 
							   unsigned short* msib_device,
							   int* _memSizeShort_device);

// headers for counting support of bitmap64
__global__ void Count64_global(unsigned int* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device);

__device__ int Count64_device(unsigned int* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device);

// DEBUG
__global__ void printBitmap();

#endif // __KERNEL_H__