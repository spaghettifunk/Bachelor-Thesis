/*
 
Copyright (c) 2004, Cornell University
All rights reserved.
 
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 
   - Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.
   - Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
   - Neither the name of Cornell University nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.
 
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
 
*/


/////////////////////////////////////////////////////////////////////
///
/// Bitmap8.cpp
///
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "Bitmap8.h"
#include "Bitmaps_cuda.h"

const int Bitmap8::_lookupTableSize = 0x10000;

int Bitmap8::_countOr = 0;
int Bitmap8::_countAnd = 0;
int Bitmap8::_countCount = 0;
int Bitmap8::_countCreateSBitmap = 0;
int* Bitmap8::_countLookupTable = 0;
int* Bitmap8::_sBitmapLookupTable = 0;
int* Bitmap8::_cBitmapLookupTable = 0;

#define DEBUG_BITMAP8COUNT

/////////////////////////////////////////////////////////////////////
/// Initialize the counting tables
/////////////////////////////////////////////////////////////////////
void Bitmap8::Init()
{
    int i, s;
    int i1, i2, a1, a2;

    // ------ initialize _countLookupTable -----------------------------
    _countLookupTable = new int[_lookupTableSize];
    memset(_countLookupTable, 0, SIZE_INT*_lookupTableSize);
    for (i = 1; i < _lookupTableSize; i++)
    {
        if (i & 0x00ff)
            _countLookupTable[i]++;
        if (i & 0xff00)
            _countLookupTable[i]++;
        assert(1 == _countLookupTable[i] || 2 == _countLookupTable[i]);
    }

    // ------ initialize _sBitmapLookupTable -----------------------------
    // note: for a customer (4 bits), set the first bit
    // (after the first bit with a one) to one
    // recall that a SHORT is 16 bits, so we need to
    // change bits for 4 customers

    int Bit8SBitmapLookUp[256];
    Bit8SBitmapLookUp[0] = 0;
    Bit8SBitmapLookUp[1] = 0;

    int curValue = 0;
    int curIndex = 1;
    for (i = 2; i < 256; i++)
    {
        if (i % curIndex == 0)
        {
            curValue = curValue + curIndex;
            curIndex *= 2;
        }
        Bit8SBitmapLookUp[i] = curValue;
    }

    _sBitmapLookupTable = new int[_lookupTableSize];
    memset(_sBitmapLookupTable, 0, SIZE_INT*_lookupTableSize);

    s = 0;  // index into the sBitmapLookupTable
    for (i1 = 0; i1 < 256; i1++)
    {
        // first customer
        a1 = Bit8SBitmapLookUp[i1] << 8;

        for (i2 = 0; i2 < 256; i2++)
        {
            // second customer
            a2 = Bit8SBitmapLookUp[i2];

            // now actually set the sBitmapLookupTable value
            _sBitmapLookupTable[s] = a1 | a2;
            s++;

        } // for i2
    } // for i1

	// GPU Variables
	int _sBitmapLookupTable8_device[_lookupTableSize];
	int _countLookupTable8_device[_lookupTableSize];

	// Memory alloc
	cudaMalloc((void**)&_sBitmapLookupTable8_device, SIZE_INT*_lookupTableSize);
	cudaMalloc((void**)&_countLookupTable8_device, SIZE_INT*_lookupTableSize);

	// copy in memory
	cudaMemcpyToSymbol("_sBitmapLookupTable8_device", &_sBitmapLookupTable[0], SIZE_INT*_lookupTableSize);
	cudaMemcpyToSymbol("_countLookupTable8_device", &_countLookupTable[0], SIZE_INT*_lookupTableSize);
}

/////////////////////////////////////////////////////////////////////
/// deallocate counting tables
/////////////////////////////////////////////////////////////////////
void Bitmap8::Destroy()
{
    if (_countLookupTable != 0)
        delete [] _countLookupTable;

    if (_sBitmapLookupTable != 0)
        delete [] _sBitmapLookupTable;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise OR 2 bitmap8s and store the result
///
/// @param b1                the first Bitmap8
/// @param b2                the second Bitmap8
/////////////////////////////////////////////////////////////////////
void Bitmap8::Or(const Bitmap8 &b1, const Bitmap8 &b2)
{
#ifdef DEBUG_BITMAP8COUNT
    _countOr++;
#endif

    // OR each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        _memory[i] = b1._memory[i] | b2._memory[i];
}

/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 bitmap8s and store the result
///
/// @param b1                the first Bitmap8
/// @param b2                the second Bitmap8
/////////////////////////////////////////////////////////////////////
void Bitmap8::And(const Bitmap8 &b1, const Bitmap8 &b2)
{

#ifdef DEBUG_BITMAP8COUNT
    _countAnd++;
#endif

	// For Candidate wise or Transaction wise technique
	// delete the comment in Bitmaps_cuda.cu

	//declare the events
	cudaEvent_t start;
	cudaEvent_t stop;
	float kernel_time;

	//create events before you use them
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// device arrays
	unsigned int* _memory_device;
	unsigned int* b1_memory;
	unsigned int* b2_memory;
	
	int b1_size = b1.getIntSize();

	// allocate memory on GPU
	cudaMalloc((void **)&b1_memory,  _memSizeInt * SIZE_UINT);
	cudaMalloc((void **)&b2_memory,  _memSizeInt * SIZE_UINT);
	cudaMalloc((void **)&_memory_device,  _memSizeInt * SIZE_UINT);

	// copy values on GPU
	cudaMemcpy(b1_memory, b1._memory, _memSizeInt * SIZE_UINT, cudaMemcpyHostToDevice );
	cudaMemcpy(b2_memory, b2._memory, _memSizeInt * SIZE_UINT, cudaMemcpyHostToDevice );
	cudaMemcpy(_memory_device, _memory, _memSizeInt * SIZE_UINT, cudaMemcpyHostToDevice );

	// need to verify it
	dim3 dimBlock(256, 1, 1);
	dim3 dimGrid(256, 1);

	//put events and kernel launches in the stream/queue
	cudaEventRecord(start,0);
	// Launch kernel 
	AndBitwiseOperation<<<dimGrid, dimBlock>>>(_memory_device, b1_size, b1_memory, b2_memory);
	cudaEventRecord(stop,0);

	//wait until the stop event is recorded
	cudaEventSynchronize(stop);

	//and get the elapsed time
	cudaEventElapsedTime(&kernel_time,start,stop);
	
	/* need to divide the elapsed time by number of blok and number of threads
	   to get the correct time of a single "function call" => it is divided by 1000
	   for getting a better visualization of the time
	 */
	// kernel_time = (kernel_time/(n_blocks * n_threads))/1000;
	// printf("Bitmap4 time taken: %n\n", kernel_time);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Wait until the kernel has finished
	cudaThreadSynchronize();

	// return values
	cudaMemcpy(_memory, _memory_device, _memSizeInt * SIZE_UINT, cudaMemcpyDeviceToHost );

	// Free Memory
	cudaFree(b1_memory);
	cudaFree(b2_memory);
	cudaFree(_memory_device);
}

/////////////////////////////////////////////////////////////////////
/// find the support of this bitmap in *number of customers*
///
/// @return the number of customers that have some bit set among their 8 bits
/////////////////////////////////////////////////////////////////////
int Bitmap8::Count()
{

#ifdef DEBUG_BITMAP8COUNT
    _countCount++;
#endif

	int support = 0;

    // we walk the memory in terms of shorts
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short
    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    
    //declare the events
	cudaEvent_t start;
	cudaEvent_t stop;
	float kernel_time;

	//create events before you use them
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// GPU Variable
	unsigned short* ms_device;
	int* support_device;
	int temp = _memSizeShort;
	int* _memSizeShort_device;

	// Memory alloc
	cudaMalloc((void**)&ms_device, sizeof(unsigned short*));
	cudaMalloc((void**)&support_device, sizeof(int*));
	cudaMalloc((void**)&_memSizeShort_device, sizeof(int*));

	// Copy in GPU
	cudaMemcpy(ms_device, ms, sizeof(unsigned short*), cudaMemcpyHostToDevice);
	cudaMemcpy(&support_device[0], &support, sizeof(int*), cudaMemcpyHostToDevice);
	cudaMemcpy(_memSizeShort_device, &temp, sizeof(int*), cudaMemcpyHostToDevice);

	// kernel parameters
	dim3 dimBlock(16, 1);
	dim3 dimGrid(1, 1, 1);

	//put events and kernel launches in the stream/queue
	cudaEventRecord(start,0);
	Count8_global<<<dimGrid, dimBlock>>>(ms_device, support_device, _memSizeShort_device);
	cudaEventRecord(stop,0);

	//wait until the stop event is recorded
	cudaEventSynchronize(stop);

	//and get the elapsed time
	cudaEventElapsedTime(&kernel_time,start,stop);
	
	/* need to divide the elapsed time by number of blok and number of threads
	   to get the correct time of a single "function call" => it is divided by 1000
	   for getting a better visualization of the time
	 */
	// kernel_time = (kernel_time/(n_blocks * n_threads))/1000;
	// printf("Bitmap4 time taken: %n\n", kernel_time);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Wait until the kernel has finished
	cudaThreadSynchronize();

	// Copy back
	cudaMemcpy(ms, ms_device, sizeof(unsigned short*), cudaMemcpyDeviceToHost);
	cudaMemcpy(&support, &support_device[0], sizeof(int*), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(ms_device);
	cudaFree(support_device);
	
	return support;
}

/////////////////////////////////////////////////////////////////////
/// Create a s-bitmap from an i-bitmap
/// <p>
/// Idea  : Again, we go thru each element of _memory. For each element,
///     if it is greater than 0, we look up the transformation table
///     (postProcessTable) to find the corresponding value for the s-bitmap,
///     and set the remaining SHORTs of the current custom to USHRT_MAX
///     (i.e. all 1's).
/// <p>
/// Note  : For example, if the bitmap is
///     [0001 | 1100 | 0011 | 1111 | 0000 | 0000] and
///     [00001111 | 00011111 | 11111111]. Refer to the paper for details.
///
/// @param iBitmap           the bitmap8 from which we create s-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap8::CreateSBitmap(const Bitmap8 &iBitmap)
{

#ifdef DEBUG_BITMAP8COUNT
    _countCreateSBitmap++;
#endif

    assert(_memory);
    assert(_memSizeShort == iBitmap._memSizeShort);

    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    const unsigned short* msib = reinterpret_cast<const unsigned short*> (iBitmap._memory);
    
    //declare the events
	cudaEvent_t start;
	cudaEvent_t stop;
	float kernel_time;

	//create events before you use them
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU Variables
	unsigned short* ms_device;
	unsigned short* msib_device;
	int _memSizeShort = iBitmap._memSizeShort;
	int* _memSizeShort_device; 

	// Memory alloc
	cudaMalloc((void**)&ms_device, sizeof(unsigned short*));
	cudaMalloc((void**)&msib_device, sizeof(const unsigned short*));
	cudaMalloc((void**)&_memSizeShort_device, sizeof(int*));

	// copy in GPU
	cudaMemcpy(ms_device, ms, sizeof(unsigned short*), cudaMemcpyHostToDevice);
	cudaMemcpy(msib_device, msib, sizeof(const unsigned short*), cudaMemcpyHostToDevice);
	cudaMemcpy(_memSizeShort_device, &_memSizeShort, sizeof(int*), cudaMemcpyHostToDevice);

	// kernel parameters
	dim3 dimBlock(256, 1);
	dim3 dimGrid(256, 1, 1);

	//put events and kernel launches in the stream/queue
	cudaEventRecord(start,0);
	CreateSBitmap8<<<dimGrid, dimBlock>>>(ms_device, msib_device, _memSizeShort_device);
	cudaEventRecord(stop,0);

	//wait until the stop event is recorded
	cudaEventSynchronize(stop);

	//and get the elapsed time
	cudaEventElapsedTime(&kernel_time,start,stop);
	
	/* need to divide the elapsed time by number of blok and number of threads
	   to get the correct time of a single "function call" => it is divided by 1000
	   for getting a better visualization of the time
	 */
	// kernel_time = (kernel_time/(n_blocks * n_threads))/1000;
	// printf("Bitmap4 time taken: %n\n", kernel_time);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// Wait until the kernel has finished
	cudaThreadSynchronize();
	
	// copy back
	cudaMemcpy(ms, ms_device, sizeof(unsigned short*), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(ms_device);
	cudaFree(msib_device);
}

