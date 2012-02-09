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
/// Bitmap32.cpp
///
/////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <limits.h>
#include "Bitmap32.h"

const int Bitmap32::_lookupTableSize = 0x10000;

int Bitmap32::_countOr = 0;
int Bitmap32::_countAnd = 0;
int Bitmap32::_countCount = 0;
int Bitmap32::_countCreateSBitmap = 0;
int Bitmap32::_countCreateCBitmap = 0;
int* Bitmap32::_sBitmapLookupTable = 0;
int* Bitmap32::_cBitmapLookupTable = 0;

#define DEBUG_BITMAP32COUNT

/////////////////////////////////////////////////////////////////////
/// Initialize the counting tables
/////////////////////////////////////////////////////////////////////
void Bitmap32::Init()
{

    int i;

    // ------ initialize sBitmapLookupTable -----------------------------
    _sBitmapLookupTable = new int[_lookupTableSize];
    memset(_sBitmapLookupTable, 0, SIZE_INT*_lookupTableSize);

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


    // ------ initialize cBitmapLookupTable -----------------------------
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
void Bitmap32::Destroy()
{
    if (_sBitmapLookupTable != 0)
        delete [] _sBitmapLookupTable;

    if (_cBitmapLookupTable != 0)
        delete [] _cBitmapLookupTable;
}

/////////////////////////////////////////////////////////////////////
/// Bitwise OR 2 bitmap32s and store the result
///
/// @param b1                the first Bitmap32
/// @param b2                the second Bitmap32
/////////////////////////////////////////////////////////////////////
void Bitmap32::Or(const Bitmap32 &b1, const Bitmap32 &b2)
{

#ifdef DEBUG_BITMAP32COUNT
    _countOr++;
#endif

    // OR each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        _memory[i] = b1._memory[i] | b2._memory[i];
}


/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 bitmaps32 and store the result
///
/// @param b1                the first Bitmap32
/// @param b2                the second Bitmap32
/////////////////////////////////////////////////////////////////////
void Bitmap32::And(const Bitmap32 &b1, const Bitmap32 &b2)
{

#ifdef DEBUG_BITMAP32COUNT
    _countAnd++;
#endif

    // AND each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        _memory[i] = b1._memory[i] & b2._memory[i];

}


/////////////////////////////////////////////////////////////////////
/// find the support of this bitmap in *number of customers*
///
/// @return the number of customers that have some bit set among their 32 bits
/////////////////////////////////////////////////////////////////////
int Bitmap32::Count()
{

#ifdef DEBUG_BITMAP32COUNT
    _countCount++;
#endif

    int support = 0;

    // we count the support
    for (int i = 0; i < _memSizeInt; i++)
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
/// @param iBitmap           the bitmap32 from which we create s-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap32::CreateSBitmap(const Bitmap32 &iBitmap)
{

#ifdef DEBUG_BITMAP32COUNT
    _countCreateSBitmap++;
#endif

    assert(_memory);
    assert(_memSizeInt == iBitmap._memSizeInt);

    // we walk the memory in terms of shorts
    // msib: pointer to memory of ibitmap in terms of short
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short
    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    const unsigned short* msib =
        reinterpret_cast<const unsigned short*> (iBitmap._memory);

    // go thru each group of 2 shorts (note that 2 shorts is one customer)
    for (int i = 0; i < _memSizeShort; i += 2)
    {
        // _sBitmapLookupTable should take in a short and return a short
        // with the first 1 changed to a
        // 0 and all remaining bits set to 1
        if (msib[i + 1] > 0)
        {
            // Post-process the first short, set the other to all 1's

            // big endian
            ms[i + 1] = _sBitmapLookupTable[msib[i + 1]];
            ms[i] = USHRT_MAX;
        }
        else
        { // Set first short to 0, post-process the second

            ms[i + 1] = 0;
            ms[i] = _sBitmapLookupTable[msib[i]];
        }
    }

}

/////////////////////////////////////////////////////////////////////
/// create a c-bitmap for compression
///
/// @param iBitmap           the bitmap32 from which we create c-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap32::CreateCBitmap(const Bitmap32 &iBitmap)
{

#ifdef DEBUG_BITMAP32COUNT
    _countCreateCBitmap++;
#endif

    assert(_memory);
    assert(_memSizeInt == iBitmap._memSizeInt);

    // we walk the memory in terms of shorts
    // msib: pointer to memory of ibitmap in terms of short
    // ms: pointer to memory as a pointer to shorts
    // ss: size of the memory in short
    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    const unsigned short* msib =
        reinterpret_cast<const unsigned short*> (iBitmap._memory);

    // go thru each group of 2 shorts (note that 2 shorts is one customer)
    for (int i = 0; i < _memSizeShort; i += 2)
    {
        // _sBitmapLookupTable should take in a short and return a short
        // with the first 1 changed to a
        // 0 and all remaining bits set to 1
        if (msib[i + 1] > 0)
        {
            // Post-process the first short, set the other to all 1's

            // big endian
            ms[i + 1] = _cBitmapLookupTable[msib[i + 1]];
            ms[i] = USHRT_MAX;
        }
        else
        { // Set first short to 0, post-process the second

            ms[i + 1] = 0;
            ms[i] = _cBitmapLookupTable[msib[i]];
        }
    }
}
