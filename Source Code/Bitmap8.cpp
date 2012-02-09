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

const int Bitmap8::_lookupTableSize = 0x10000;

int Bitmap8::_countOr = 0;
int Bitmap8::_countAnd = 0;
int Bitmap8::_countCount = 0;
int Bitmap8::_countCreateSBitmap = 0;
int Bitmap8::_countCreateCBitmap = 0;
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


    // ------ initialize cBitmapLookupTable -----------------------------

    int Bit8CBitmapLookUp[256];
    Bit8CBitmapLookUp[0] = 0;
    Bit8CBitmapLookUp[1] = 1;

    curValue = 1;
    curIndex = 1;
    for (i = 2; i < 256; i++)
    {
        if (i % curIndex == 0)
        {
            curIndex *= 2;
            curValue = curValue + curIndex;
        }
        Bit8CBitmapLookUp[i] = curValue;
    }

    _cBitmapLookupTable = new int[_lookupTableSize];
    memset(_cBitmapLookupTable, 0, SIZE_INT*_lookupTableSize);


    s = 0;  // index into the sBitmapLookupTable
    // these are 16 bits: 4 customers
    for (i1 = 0; i1 < 256; i1++)
    {
        // first customer
        a1 = Bit8CBitmapLookUp[i1] << 8;

        for (i2 = 0; i2 < 256; i2++)
        {
            // second customer
            a2 = Bit8CBitmapLookUp[i2];

            // now actually set the sBitmapLookupTable value
            _cBitmapLookupTable[s] = a1 | a2;
            s++;

        } // for i2
    } // for i1


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

    if (_cBitmapLookupTable != 0)
        delete [] _cBitmapLookupTable;
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

    // OR each int bitwise
    for (int i = 0; i < b1._memSizeInt; i++)
        _memory[i] = b1._memory[i] & b2._memory[i];
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


    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    int support = 0;

    // we count the support
    // go thru the whole bitmap in steps of SHORT (but note that each
    // customer uses 4 bits)
    for (int i = 0; i < _memSizeShort; i++)
        support += _countLookupTable[ms[i]];

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
    assert(_memSizeInt == iBitmap._memSizeInt);

    // we walk the memory in terms of chars
    // msib: pointer to memory of ibitmap in terms of char
    // ms: pointer to memory as a pointer to char
    // ss: size of the memory in char
    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    const unsigned short* msib =
        reinterpret_cast<const unsigned short*> (iBitmap._memory);

    // go thru each SHORT (note that a SHORT is 4 customers)
    for (int i = 0; i < _memSizeShort; i++)
        ms[i] = _sBitmapLookupTable[msib[i]];
}

/////////////////////////////////////////////////////////////////////
/// create a c-bitmap for compression
///
/// @param iBitmap           the bitmap8 from which we create c-bitmap
/////////////////////////////////////////////////////////////////////
void Bitmap8::CreateCBitmap(const Bitmap8 &iBitmap)
{

#ifdef DEBUG_BITMAP8COUNT
    _countCreateCBitmap++;
#endif

    assert(_memory);
    assert(_memSizeInt == iBitmap._memSizeInt);

    // we walk the memory in terms of chars
    // msib: pointer to memory of ibitmap in terms of char
    // ms: pointer to memory as a pointer to char
    // ss: size of the memory in char
    unsigned short* ms = reinterpret_cast<unsigned short*>(_memory);
    const unsigned short* msib =
        reinterpret_cast<const unsigned short*> (iBitmap._memory);

    // go thru each SHORT (note that a SHORT is 4 customers)
    for (int i = 0; i < _memSizeShort; i++)
        ms[i] = _cBitmapLookupTable[msib[i]];
}

