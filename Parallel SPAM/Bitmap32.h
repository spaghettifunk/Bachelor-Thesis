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
/// Bitmap32.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __BITMAP32_H__
#define __BITMAP32_H__


#include <assert.h>
#include <memory.h>
#include "Tables.h"

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessingGroup Bitmap Processing
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A vertical representation of the data. It handles both uncompressed
/// and compressed data.
/// <p>
/// This bitmap can represent customers with up to 32 transactions since it
///  allocates exactly 32 bits per customer
/////////////////////////////////////////////////////////////////////
class Bitmap32
{
    friend class SeqBitmap;

public:

     static void Init();
     static void Destroy();

     static int CalcSize (int numCust)
    {
        return numCust * SIZE_UINT;
    }

    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for Bitmap32 for this many customers
    ///
    /// @param numCustomers  # of customers
    /// @param memory
    /////////////////////////////////////////////////////////////////////
     Bitmap32(int numCustomers, unsigned int *&memory)
    {
        _numCust = numCustomers;
        _memSizeShort = _numCust * SHORT_PER_INT;
        _memSizeInt = _numCust;
        _memory = memory;
        memory += _memSizeInt * SIZE_UINT;

        memset(_memory, 0, _memSizeInt * SIZE_UINT);
    };


    /////////////////////////////////////////////////////////////////////
    /// Copy constructor
    ///
    /// @param b             Bitmap32 to copy
    /// @param memory
    /////////////////////////////////////////////////////////////////////
    Bitmap32(Bitmap32 &b, unsigned int *&memory)
    {
        // copy information from the source bitmap
        _memSizeShort = b._memSizeShort;
        _memSizeInt = b._memSizeInt;
        _numCust = b._numCust;
        _memory = memory;
        memory += _memSizeInt * SIZE_UINT;

        memcpy(_memory, b._memory, _memSizeInt * SIZE_UINT);
    };


    /////////////////////////////////////////////////////////////////////
    /// Destructor (does nothing)
    /////////////////////////////////////////////////////////////////////
     ~Bitmap32()
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
    /// @param j             bit position in Bitmap to be changed
    /////////////////////////////////////////////////////////////////////
     void FillEmptyPosition(int j)
    {
        // Locate correct int in the Bitmap
        int i = j / BITS_PER_INT ;

        // switch on correct bit
        _memory[i] = _memory[i] | Bit32Table[(j % BITS_PER_INT )];
    };


    // --- anding and oring two bitmap32s
     void Or(const Bitmap32 &b1, const Bitmap32 &b2);
     void And(const Bitmap32 &b1, const Bitmap32 &b2);

    // count the number of customers
     int Count();
     void CreateSBitmap(const Bitmap32 &iBitmap);
     void CreateCBitmap(const Bitmap32 &iBitmap);


    // ----------------------- static variables ----------------------
    /// counters for performance measurements
    static int _countOr;
    static int _countAnd;
    static int _countCount;
    static int _countCreateSBitmap;

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
    unsigned int* _memory;      ///< where data is stored
    int _memSizeInt;            ///< size of memory allocated in sizeof(int)
    int _memSizeShort;          ///< size of memory allocated in sizeof(short)
    int _numCust;               ///< number of customers

}
; // class Bitmap32

/** @} */
#endif
