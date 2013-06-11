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
/// Bitmap16.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __BITMAP16_H__
#define __BITMAP16_H__

#include <assert.h>
#include <memory.h>
#include "Tables.h"

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessingGroup Bitmap Processing
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A vertical representation of the data. It handles both uncompressed
///  and compressed data.
///  <p>
/// This bitmap can represent customers with up to 16 transactions since it
///  allocates exactly 16 bits per customer
/////////////////////////////////////////////////////////////////////
class Bitmap16
{
    friend class SeqBitmap;

public:

     static void Init();
     static void Destroy();

     static int CalcSize (int numCust)
    {
        if (numCust % 2 == 0)
        {
            return (numCust / 2) * SIZE_USHORT;
        }
        else
            return (numCust / 2 + 1) * SIZE_USHORT;
    }

    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for Bitmap16 for this many customers
    ///
    /// @param numCustomers  # of customers
    /// @param memory
    /////////////////////////////////////////////////////////////////////
     Bitmap16(int numCustomers, unsigned short*& memory)
    {
        _numCust = numCustomers;

        if (_numCust % 2 == 0)
            _memSizeInt = _numCust / 2;
        else
            _memSizeInt = _numCust / 2 + 1;

        _memSizeShort = _memSizeInt * SHORT_PER_INT;
        _memory = memory;
        memory += _memSizeShort * SIZE_USHORT;
        memset(_memory, 0, _memSizeShort * SIZE_USHORT);
    };

    /////////////////////////////////////////////////////////////////////
    /// Copy constructor
    ///
    /// @param b             Bitmap to copy
    /// @param memory
    /////////////////////////////////////////////////////////////////////
    __host__ __device__ Bitmap16(Bitmap16 &b, unsigned short *&memory)
    {
        // copy information from the source bitmap
        _memSizeShort = b._memSizeShort;
        _memSizeInt = b._memSizeInt;
        _numCust = b._numCust;
        _memory = memory;
        memory += _memSizeShort * SIZE_USHORT;
        memcpy(_memory, b._memory, _memSizeShort * SIZE_USHORT);
    };

    /////////////////////////////////////////////////////////////////////
    /// Destructor (does nothing)
    /////////////////////////////////////////////////////////////////////
     ~Bitmap16()
    {
        //if (_memory)
        // delete [] _memory;
    };

    /////////////////////////////////////////////////////////////////////
    /// Pop this bitmap's memory from the global stack (allocation is done in
    /// the constructor, but deallocation is not done in the destructor
    /////////////////////////////////////////////////////////////////////
     void Deallocate(unsigned short *&memory)
    {
        memory -= (_memSizeShort * SIZE_USHORT);
    };

    /////////////////////////////////////////////////////////////////////
    /// Fill in a 1 in a certain position
    ///
    /// @param j             bit position in Bitmap to be changed
    /////////////////////////////////////////////////////////////////////
     void FillEmptyPosition(int j)
    {
        // Locate correct short in the bitmap
        int i = j / BITS_PER_SHORT;

        // switch on correct bit
        _memory[i] = _memory[i] | Bit16Table[(j % BITS_PER_SHORT )];
    };

    // --- anding and oring two bitmap16s
     void Or(const Bitmap16 &b1, const Bitmap16 &b2);
     void And(const Bitmap16 &b1, const Bitmap16 &b2);

    // count the number of customers
     int Count();
     void CreateSBitmap(const Bitmap16 &iBitmap);

    // ----------------------- static variables ----------------------
    /// counters for performance measurements
    static int _countOr;
    static int _countAnd;
    static int _countCount;
    static int _countCreateSBitmap;
    static int _countCreateCBitmap;

    /// lookup table for counting
    static const int _lookupTableSize;

    /// lookup table for creating a s-bitmap from an i-bitmap
    static int* _sBitmapLookupTable;

    /// lookup table for creating a c-bitmap for compression
    static int* _cBitmapLookupTable;

     int getIntSize() const
    {
        return _memSizeInt;
    }

    // --- private variables
    unsigned short* _memory;    ///< where data is stored
    int _memSizeInt;            ///< size of memory allocated in sizeof(int)
    int _memSizeShort;          ///< size of memory allocated in sizeof(short)
    int _numCust;               ///< number of customers

}
; // class Bitmap16

/** @} */
#endif
