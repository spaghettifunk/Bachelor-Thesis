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
/// Bitmap16.cpp
///
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include "Bitmap16.h"

const int Bitmap16::_lookupTableSize = 0x10000;

int Bitmap16::_countOr = 0;
int Bitmap16::_countAnd = 0;
int Bitmap16::_countCount = 0;
int Bitmap16::_countCreateSBitmap = 0;
int Bitmap16::_countCreateCBitmap = 0;
int* Bitmap16::_sBitmapLookupTable = 0;
int* Bitmap16::_cBitmapLookupTable = 0;

#define DEBUG_BITMAP16COUNT

/////////////////////////////////////////////////////////////////////
/// Initialize the counting tables
/////////////////////////////////////////////////////////////////////
void Bitmap16::Init()
{

    int i;

    // ------ initialize sBitmapLookupTable -----------------------------
    _sBitmapLookupTable = new int[_lookupTableSize];
    memset(_sBitmapLookupTable, 0, SIZE_UINT*_lookupTableSize);

    _sBitmapLookupTable[0] = 0;
    _sBitmapLookupTable[1] = 0;

    int curValue = 0;
    int curIndex = 1;
    for (i = 2; i < _lookupTableSize; i++)
    {
        if (i % curIndex == 0)
        {
            curValue = curValue + curIndex;
            curIndex *= 2;
        }
        _sBitmapLookupTable[i] = curValue;
    }


    // ------ initialize sBitmapLookupTable -----------------------------
    _cBitmapLookupTable = new int[_lookupTableSize];
    memset(_cBitmapLookupTable, 0, SIZE_UINT*_lookupTableSize);

    _cBitmapLookupTable[0] = 0;
    _cBitmapLookupTable[1] = 1;

    curValue = 1;
    curIndex = 1;
    for (i = 2; i < _lookupTableSize; i++)
    {
        if (i % curIndex == 0)
        {
            curIndex *= 2;
            curValue = curValue + curIndex;
        }
        _cBitmapLookupTable[i] = curValue;
    }
}

/////////////////////////////////////////////////////////////////////
/// deallocate counting tables
/////////////////////////////////////////////////////////////////////
void Bitmap16::Destroy()
{
    if (_sBitmapLookupTable != 0)
        delete [] _sBitmapLookupTable;

    if (_cBitmapLookupTable != 0)
        delete [] _cBitmapLookupTable;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise OR 2 bitmaps and store the result
///
/// @param b1                the first Bitmap16
/// @param b2                the second Bitmap16
/////////////////////////////////////////////////////////////////////
void Bitmap16::Or(const Bitmap16 &b1, const Bitmap16 &b2)
{

#ifdef DEBUG_BITMAP16COUNT
    _countOr++;
#endif

    unsigned int* const b1ptr =
        reinterpret_cast<unsigned int * const>(b1._memory);
    unsigned int* const b2ptr =
        reinterpret_cast<unsigned int * const>(b2._memory);
    unsigned int* memory =
        reinterpret_cast<unsigned int *>(_memory);

    // AND each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        memory[i] = b1ptr[i] | b2ptr[i];
}


/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 bitmaps and store the result
///
/// @param b1                the first Bitmap16
/// @param b2                the second Bitmap16
/////////////////////////////////////////////////////////////////////
void Bitmap16::And(const Bitmap16 &b1, const Bitmap16 &b2)
{
#ifdef DEBUG_BITMAP16COUNT
    _countAnd++;
#endif

    // AND each short bitwise
    unsigned int* const b1ptr =
        reinterpret_cast<unsigned int * const>(b1._memory);
    unsigned int* const b2ptr =
        reinterpret_cast<unsigned int * const>(b2._memory);
    unsigned int* memory = reinterpret_cast<unsigned int *>(_memory);

    // AND each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        memory[i] = b1ptr[i] & b2ptr[i];
}


/////////////////////////////////////////////////////////////////////
/// find the support of this bitmap in *number of customers*
///
/// @return the number of customers that have some bit set among their 16 bits
/////////////////////////////////////////////////////////////////////
int Bitmap16::Count()
{

#ifdef DEBUG_BITMAP16COUNT
    _countCount++;
#endif

    int support = 0;

    // we count the support
    // go thru the whole bitmap in steps of SHORT
    for (int i = 0; i < _memSizeShort; i++)
        if (_memory[i] > 0)
            support++;

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
/// @param iBitmap           the bitmap from which we create s-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap16::CreateSBitmap(const Bitmap16 &iBitmap)
{

#ifdef DEBUG_BITMAP16COUNT
    _countCreateSBitmap++;
#endif

    assert(_memory);
    assert(_memSizeShort == iBitmap._memSizeShort);

    // go thru each SHORT
    for (int i = 0; i < _memSizeShort; i++)
        _memory[i] = _sBitmapLookupTable[iBitmap._memory[i]];

}

/////////////////////////////////////////////////////////////////////
/// create a c-bitmap for compression
///
/// @param iBitmap           the bitmap16 from which we create c-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap16::CreateCBitmap(const Bitmap16 &iBitmap)
{

#ifdef DEBUG_BITMAP16COUNT
    _countCreateCBitmap++;
#endif

    assert(_memory);
    assert(_memSizeShort == iBitmap._memSizeShort);

    // go thru each SHORT
    for (int i = 0; i < _memSizeShort; i++)
        _memory[i] = _cBitmapLookupTable[iBitmap._memory[i]];

}

