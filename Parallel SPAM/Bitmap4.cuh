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
/// Bitmap4.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __BITMAP4_H__
#define __BITMAP4_H__
 
#include <assert.h>
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "Tables.h"

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessingGroup Bitmap Processing
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A vertical representation of the data. It handles both uncompressed
///  and compressed data.
///  <p>
/// This bitmap can represent customers with up to 4 transactions since it
///  allocates exactly 4 bits per customer
/////////////////////////////////////////////////////////////////////
class Bitmap4
{
    friend class SeqBitmap;

public:

     static void Init();
     static void Destroy();

     static int CalcSize(int numCust)
    {
        // calculate the total number of bits
        int n = (4 * numCust);

        // round the size in bits up to a multiple of BITS_PER_INT
        int k = n % BITS_PER_INT;
        if (k)
        {
            n += BITS_PER_INT - k;
        }

        return (n / BITS_PER_INT) * SIZE_UINT;
    }


    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for Bitmap4 for this many customers
    ///
    /// @param numCustomers  # of customers
    /// @param memory
    /////////////////////////////////////////////////////////////////////
    Bitmap4(int numCustomers, unsigned int *&memory)
    {
        // calculate the total number of bits
        int n = (4 * numCustomers);

        // round the size in bits up to a multiple of BITS_PER_INT
        int k = n % BITS_PER_INT;
        if (k)
        {
            n += BITS_PER_INT - k;
        }

        assert(0 == (n % BITS_PER_INT));

        // translate this to a number of ints
        _numCust = numCustomers;
        _memSizeInt = n / BITS_PER_INT;
        _memSizeShort = _memSizeInt * SHORT_PER_INT;

        // Push this bitmap onto the stack
        _memory = memory;

        // Update the global stack pointer
        memory += _memSizeInt * SIZE_UINT;

        //_memory = new unsigned int[_memSizeInt];
        memset(_memory, 0, _memSizeInt * SIZE_UINT);

    };

    /////////////////////////////////////////////////////////////////////
    /// Copy constructor
    ///
    /// @param b             Bitmap4 to copy
    /// @param memory
    /////////////////////////////////////////////////////////////////////
    Bitmap4(Bitmap4 &b, unsigned int *&memory):
            _memSizeInt(b._memSizeInt),
            _memSizeShort(b._memSizeShort),
            _numCust(b._numCust)
    {

        // copy information from the source bitmap
        _memory = memory;

        // Update the global stack pointer
        memory += _memSizeInt * SIZE_UINT;
        memcpy(_memory, b._memory, _memSizeInt * SIZE_UINT);
    };

    /////////////////////////////////////////////////////////////////////
    /// Destructor (does nothing)
    /////////////////////////////////////////////////////////////////////
    ~Bitmap4()
    { };

    /////////////////////////////////////////////////////////////////////
    /// Pop this bitmap's memory from the global stack (allocation is done in
    /// the constructor, but deallocation is not done in the destructor
    /////////////////////////////////////////////////////////////////////
    void Deallocate(unsigned int *&memory)
    {
        memory -= (_memSizeInt * SIZE_UINT);
    };

    /////////////////////////////////////////////////////////////////////
    /// Fill in a 1 in a certain position
    ///
    /// @param j             bit position in Bitmap4 to be changed
    /////////////////////////////////////////////////////////////////////
     void FillEmptyPosition(int j)
    {
        // Locate correct int in the Bitmap
        int i = j / BITS_PER_INT ;

        // switch on correct bit
        _memory[i] = _memory[i] | Bit32Table[(j % BITS_PER_INT )];
    };


    // anding and oring two bitmap4s
     void Or(const Bitmap4 &b1, const Bitmap4 &b2);
     void And(const Bitmap4 &b1, const Bitmap4 &b2);

    // count the number of customers
     int Count();

    // create bitmaps for specific purpose
    void CreateSBitmap(const Bitmap4 &iBitmap);

     int getIntSize() const
    {
        return _memSizeInt;
    }

    // ----------------------- static variables ----------------------
    /// counters for performance measurements
    static int _countOr;
    static int _countAnd;
    static int _countCount;
    static int _countCreateSBitmap;

    /// lookup table for counting
    static const int _lookupTableSize;

    /// lookup table for the counting
    static int* _countLookupTable;

    /// lookup table for creating a s-bitmap from an i-bitmap
    static int* _sBitmapLookupTable;

    /// lookup table for creating a c-bitmap for compression
    static int* _cBitmapLookupTable;

    // --- private variables
    unsigned int* _memory;      ///< where data is stored
    int _memSizeInt;            ///< size of memory allocated in sizeof(int)
    int _memSizeShort;          ///< size of memory allocated in sizeof(short)
    int _numCust;               ///< number of customers

}
; // class Bitmap4

/** @} */
#endif
