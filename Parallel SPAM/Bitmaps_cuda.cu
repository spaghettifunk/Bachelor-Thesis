#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include "vs2010_fix_cuda.h"
#include "SeqBitmap.h"
#include "Bitmaps_cuda.h"

//#define __PRINT_DEBUG__

#define N 65536
#define BIG_ENDIAN 0xffff

//#define __TRANSACTION_WISE__
//#define __CANDIDATE_WISE__

// Bitmap4
__device__ int _sBitmapLookupTable4_device[N];
__device__ int _countLookupTable4_device[N];

// Bitmap8
__device__ int _sBitmapLookupTable8_device[N];
__device__ int _countLookupTable8_device[N];

// Bitmap16
__device__ int _sBitmapLookupTable16_device[N];

// Bitmap32
__device__ int _sBitmapLookupTable32_device[N];

// Bitmap64
__device__ int _sBitmapLookupTable64_device[N];


// REDUCTION!!
template <unsigned int blockSize>
__device__ void reduce(int *g_idata, int *g_odata, unsigned int n)
{

	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;

	sdata[tid] = 0;

	while (i < n) 
	{ 
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];  
		i += gridSize; 
	}
	__syncthreads();
	
	if (blockSize >= 512) { 
		if (tid < 256) { 
			sdata[tid] += sdata[tid + 256]; 
		} __syncthreads(); 
	}
	if (blockSize >= 256) { 
		if (tid < 128) { 
			sdata[tid] += sdata[tid + 128]; 
		} 
		__syncthreads(); 
	}
	if (blockSize >= 128) { 
		if (tid <  64) { 
			sdata[tid] += sdata[tid +  64];
		}
		__syncthreads(); 
	}
	
	if (tid < 32) 
	{
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
	}

	if (tid == 0) 
		g_odata[blockIdx.x] = sdata[0];
}

#ifdef __PRINT_DEBUG__
// Printing Bitmaps for Debugging
__global__ void printBitmap(){
    
	//get unique block index
    int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D

    //get unique thread index
    int threadId = blockId * blockDim.x + threadIdx.x; 	

	if(threadId >= N)
		return;

	__syncthreads();
	printf("_sBitmap[%d]: %d\n", threadId, /*_sBitmapLookupTable4_device[threadId]*/);
}
#endif

//	Kernel functions for AND and OR Bitwise operations - common for all Bitmaps
// Transaction-wise (need to test with Candidate-wise)
__global__ void AndBitwiseOperation(unsigned int* _memory_device,  
									const int b1_size, 
									unsigned int* b1_memory, 
									unsigned int* b2_memory)
{
	// Candidate wise technique for ANDing operation
#ifdef __CANDIDATE_WISE__
	int i = threadIdx.x;
	
	while (i < b1_size){
		_memory_device[i] = b1_memory[i] & b2_memory[i];
		i += blockDim.x;
	}

	// global reduce
	// works with small dataset - Registers issues for sure
	reduce<256>(_memory_device, _memory_device, n);

	// after the ANDing there is the candidate generation (counting op)!

#endif

	// Transaction wise technique for ANDing operation
#ifdef __TRANSACTION_WISE__
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int n = b1_size;

	while (i < b1_size){
		_memory_device[i] = b1_memory[i] & b2_memory[i];
		i += blockDim.x * gridDim.x;
	}

	// global reduce
	// works with small dataset - Registers issues for sure	
	reduce<256>(_memory_device, _memory_device, n);

	// after the ANDing there is the candidate generation (counting op)!

#else
	// Simple ANDing opeartion using CUDA

    //get unique block index
    int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D

    //get unique thread index
    int threadId = blockId * blockDim.x + threadIdx.x; 

    //check global unique thread range
    if(threadId >= b1_size)
        return;

    //output bitwise and
    _memory_device[threadId] = b1_memory[threadId] & b2_memory[threadId];

#endif
}

// BITMAP4
__global__ void CreateSBitmap4(unsigned short* ms_device, 
							   unsigned short* msib_device, 
							   int* _memSizeShort_device)
{
    // we walk the memory in terms of shorts
    // msib: pointer to memory of ibitmap in terms of short
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short

    //get unique block index
    int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D

    //get unique thread index
    int threadId = blockId * blockDim.x + threadIdx.x; 

    //check global unique thread range
    if(threadId >= _memSizeShort_device[0])
        return;
	
	// go thru each SHORT (note that a SHORT is 4 customers)
	ms_device[threadId] = _sBitmapLookupTable4_device[msib_device[threadId]];
}

__global__ void Count4_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device)
{
	Count4_device(ms_device, support_device, _memSizeShort_device);
}

__device__ int Count4_device(unsigned short* ms_device, 
							 int* support_device,
							 int* _memSizeShort_device)
{

    // we count the support
    // go thru the whole bitmap in steps of SHORT (but note that each
    // customer uses 4 bits)
	int support = 0;

	int i = threadIdx.x;
	
	while(i < _memSizeShort_device[0])
		support += _countLookupTable4_device[ms_device[i]];

	support_device[0] = support;

    return *support_device;
}

// BITMAP8
__global__ void CreateSBitmap8(unsigned short* ms_device, 
							   unsigned short* msib_device, 
							   int* _memSizeShort_device)
{
    // we walk the memory in terms of shorts
    // msib: pointer to memory of ibitmap in terms of short
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short

    //get unique block index
    int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D

    //get unique thread index
    int threadId = blockId * blockDim.x + threadIdx.x; 

    //check global unique thread range
    if(threadId >= _memSizeShort_device[0])
        return;
	
	// go thru each SHORT (note that a SHORT is 4 customers)
	ms_device[threadId] = _sBitmapLookupTable8_device[msib_device[threadId]];
}

__global__ void Count8_global(unsigned short* ms_device, 
							  int* support_device, 
							  int* _memSizeShort_device)
{
	Count8_device(ms_device, support_device, _memSizeShort_device);
}

__device__ int Count8_device(unsigned short* ms_device, 
							 int* support_device, 
							 int* _memSizeShort_device)
{
    // we count the support
    // go thru the whole bitmap in steps of SHORT (but note that each
    // customer uses 4 bits)
	int support = 0;

	int i = threadIdx.x;
	
	while(i < _memSizeShort_device[0])
		support += _countLookupTable8_device[ms_device[i]];

	support_device[0] = support;

    return *support_device;
}

// BITMAP16
__global__ void CreateSBitmap16(unsigned short* ms_device, 
							   unsigned short* msib_device, 
							   int* _memSizeShort_device)
{
	// NEED TO TEST WITH BIG DATASETS!!

    // we walk the memory in terms of shorts
    // msib: pointer to memory of ibitmap in terms of short
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short

    //get unique block index
    int blockId = blockIdx.x //1D
        + blockIdx.y * gridDim.x //2D
        + gridDim.x * gridDim.y * blockIdx.z; //3D

    //get unique thread index
    int threadId = blockId * blockDim.x + threadIdx.x; 

    //check global unique thread range
    if(threadId >= _memSizeShort_device[0])
        return;
	
	// go thru each SHORT (note that a SHORT is 4 customers)
	ms_device[threadId] = _sBitmapLookupTable16_device[msib_device[threadId]];
}

__global__ void Count16_global(unsigned short* ms_device, 
							  int* support_device, 
							  int* _memSizeShort_device)
{
	Count16_device(ms_device, support_device, _memSizeShort_device);
}

__device__ int Count16_device(unsigned short* _memory_device, 
							  int* support_device, 
							  int* _memSizeShort_device)
{
	int support[N];

	int n = _memSizeShort_device[0];

	int i = threadIdx.x;

    // we count the support
    // go thru the whole bitmap in steps of SHORT
    while(i < _memSizeShort_device[0])
        if (_memory_device[i] > 0)
            support[i] = 1;

	// reduction
	// works with small dataset - Registers issues for sure
	reduce<256>(support_device, support, n);

    return *support_device;
}

// BITMAP32
__global__ void CreateSBitmap32(unsigned short* ms_device, 
							    unsigned short* msib_device,
							    int* _memSizeShort_device)
{
	int i = threadIdx.x;

    // go thru each group of 2 shorts (note that 2 shorts is one customer)
	for (i = 0; i < _memSizeShort_device[0]; i += blockDim.x*blockIdx.x) // i += 2
    {
        // _sBitmapLookupTable32 should take in a short and return a short
        // with the first 1 changed to a
        // 0 and all remaining bits set to 1
        if (msib_device[i + 1] > 0)
        {
            // Post-process the first short, set the other to all 1's
            // big endian
            ms_device[i + 1] = _sBitmapLookupTable32_device[msib_device[i + 1]];
            ms_device[i] = BIG_ENDIAN;
        }
        else
        { 
			// Set first short to 0, post-process the second
            ms_device[i + 1] = 0;
            ms_device[i] = _sBitmapLookupTable32_device[msib_device[i]];
        }
    }
}

// we can call the same function as Bitmap16
__global__ void Count32_global(unsigned short* ms_device, 
							  int* support_device,
							  int* _memSizeShort_device)
{
	// can call the same kernel function!!
	Count16_device(ms_device, support_device, _memSizeShort_device);
}

// BITMAP64
__global__ void CreateSBitmap64(unsigned short* ms_device, 
							   unsigned short* msib_device, 
							   int* _memSizeShort_device)
{
	int i = threadIdx.x;

    // go thru each group of 4 shorts (note that 4 shorts is one customer)
	for (i = 0; i < _memSizeShort_device[0]; i += blockDim.x*blockIdx.x) // i += 4
    {
        // _sBitmapLookupTable should take in a short and return a short
        // with the first 1 changed to a
        // 0 and all remaining bits set to 1
        if (msib_device[i + 1] > 0)
        {
            // Post-process the first short, set the others to all 1's
            ms_device[i] = USHRT_MAX;
            ms_device[i + 1] = _sBitmapLookupTable64_device[msib_device[i + 1]];
            ms_device[i + 2] = USHRT_MAX;
            ms_device[i + 3] = USHRT_MAX;
        }
        else if (msib_device[i] > 0)
        {
            ms_device[i] = _sBitmapLookupTable64_device[msib_device[i]];
            ms_device[i + 1] = 0;
            ms_device[i + 2] = USHRT_MAX;
            ms_device[i + 3] = USHRT_MAX;
        }
        else if (msib_device[i + 3] > 0)
        {
            ms_device[i] = 0;
            ms_device[i + 1] = 0;
            ms_device[i + 2] = USHRT_MAX;
            ms_device[i + 3] = _sBitmapLookupTable64_device[msib_device[i + 3]];
        }
        else
        { // Set first 3 shorts to 0, post-process the last short
            ms_device[i] = 0;
            ms_device[i + 1] = 0;
            ms_device[i + 2] = _sBitmapLookupTable64_device[msib_device[i + 2]];
            ms_device[i + 3] = 0;
        }
    }
}

__global__ void Count64_global(unsigned int* ms_device, 
							  int* support_device, 
							  int* _memSizeShort_device)
{
	Count64_device(ms_device, support_device, _memSizeShort_device);
}

__device__ int Count64_device(unsigned int* _memory_device, 
							  int* support_device, 
							  int* _memSizeInt_device)
{
    int support[N];

	int n = _memSizeInt_device[0];

	int i = threadIdx.x;

    // Go through each group of 2 ints
    for (i = 0; i < _memSizeInt_device[0]; i += blockDim.x*blockIdx.x) // i += 2
        if ((_memory_device[i] > 0) || (_memory_device[i + 1] > 0))
            support[i] = 1;

	// local reduction
	// works with small dataset - Registers issues for sure
	reduce<256>(support_device, support, n);

    return *support_device;
}

