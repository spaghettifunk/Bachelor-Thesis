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
/// SeqBitmap.cpp
///
/////////////////////////////////////////////////////////////////////

#include <ostream>
#include <fstream>
#include <stdio.h>
#include "SeqBitmap.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

using namespace std;

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessingGroup
/** @{ */

/// number of different bitmaps in SeqBitmap
const int NUM_BITMAP = 5;

/// the size of each customer data in each bitmap
const int BITMAP_LENGTH[5] =
    {
        4, 8, 16, 32, 64
    };

// Variables used to gather usage statistics
int SeqBitmap::_countOr = 0;
int SeqBitmap::_countAnd = 0;
int SeqBitmap::_countCount = 0;
int SeqBitmap::_countCreateSBitmap = 0;

unsigned int* SeqBitmap::_memory4 = 0;
unsigned int* SeqBitmap::_memory8 = 0;
unsigned short* SeqBitmap::_memory16 = 0;
unsigned int* SeqBitmap::_memory32 = 0;
unsigned int* SeqBitmap::_memory64 = 0;

// Bitmap sizes within this SeqBitmap
int SeqBitmap::_size4 = 0;
int SeqBitmap::_size8 = 0;
int SeqBitmap::_size16 = 0;
int SeqBitmap::_size32 = 0;
int SeqBitmap::_size64 = 0;

#define DEBUG_SEQBITMAPCOUNT

/////////////////////////////////////////////////////////////////////
/// Initialize the lookup tables to be used by this SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Init()
{ 
	Bitmap4::Init();
    Bitmap8::Init();
    Bitmap16::Init();
    Bitmap32::Init();
    Bitmap64::Init();
}

/////////////////////////////////////////////////////////////////////
/// Initialize all SeqBitmap memory for duration of program
///
/// @param size4             total space required for all bitmap_4's
/// @param size8             total space required for all bitmap_8's
/// @param size16            total space required for all bitmap_16's
/// @param size32            total space required for all bitmap_32's
/// @param size64            total space required for all bitmap_64's
/////////////////////////////////////////////////////////////////////
void SeqBitmap::MemAlloc(
    int size4,
    int size8,
    int size16,
    int size32,
    int size64)
{

    _size4 = size4;
    _size8 = size8;
    _size16 = size16;
    _size32 = size32;
    _size64 = size64;

    _memory4 = new unsigned int[size4];
    _memory8 = new unsigned int[size8];
    _memory16 = new unsigned short[size16];
    _memory32 = new unsigned int[size32];
    _memory64 = new unsigned int[size64];
}

/////////////////////////////////////////////////////////////////////
/// Delete all of SeqBitmap's memory
/////////////////////////////////////////////////////////////////////
void SeqBitmap::MemDealloc()
{
	//printf("SeqBitmap::Dealloc\n\n\n");

    delete [] _memory4;
    delete [] _memory8;
    delete [] _memory16;
    delete [] _memory32;
    delete [] _memory64;
}

void SeqBitmap::Destroy()
{
	//printf("SeqBitmap::Destroy\n\n\n");

    Bitmap4::Destroy();
    Bitmap8::Destroy();
    Bitmap16::Destroy();
    Bitmap32::Destroy();
    Bitmap64::Destroy();
}

/////////////////////////////////////////////////////////////////////
/// Bitwise OR 2 SeqBitmaps and store the result
///
/// @param b1                the first SeqBitmap
/// @param b2                the second SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Or(const SeqBitmap &b1, const SeqBitmap &b2)
{

#ifdef DEBUG_SEQBITMAPCOUNT
    _countOr++;
#endif

    // OR all sizes of bitmaps within this SeqBitmap
    if (b1._bitmap4 != 0)
        _bitmap4->Or(*b1._bitmap4, *b2._bitmap4);

    if (b1._bitmap8 != 0)
        _bitmap8->Or(*b1._bitmap8, *b2._bitmap8);

    if (b1._bitmap16 != 0)
        _bitmap16->Or(*b1._bitmap16, *b2._bitmap16);

    if (b1._bitmap32 != 0)
        _bitmap32->Or(*b1._bitmap32, *b2._bitmap32);

    if (b1._bitmap64 != 0)
        _bitmap64->Or(*b1._bitmap64, *b2._bitmap64);
}


/////////////////////////////////////////////////////////////////////
/// Bitwise AND 2 SeqBitmaps and store the result
///
/// @param b1                the first SeqBitmap
/// @param b2                the second SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::And(const SeqBitmap &b1, const SeqBitmap &b2)
{
#ifdef DEBUG_SEQBITMAPCOUNT
    _countAnd++;
#endif

    // AND all bitmap sizes stored within this SeqBitmap
    if (b1._bitmap4 != 0)
        _bitmap4->And(*b1._bitmap4, *b2._bitmap4);

    if (b1._bitmap8 != 0)
        _bitmap8->And(*b1._bitmap8, *b2._bitmap8);

    if (b1._bitmap16 != 0)
        _bitmap16->And(*b1._bitmap16, *b2._bitmap16);

    if (b1._bitmap32 != 0)
        _bitmap32->And(*b1._bitmap32, *b2._bitmap32);

    if (b1._bitmap64 != 0)
        _bitmap64->And(*b1._bitmap64, *b2._bitmap64);
}

/////////////////////////////////////////////////////////////////////
/// find the support of this bitmap
/////////////////////////////////////////////////////////////////////
int SeqBitmap::Count()
{
#ifdef DEBUG_SEQBITMAPCOUNT
    _countCount++;
#endif

    int count = 0;

    // Add the counts together for all of the bitmap sizes
    if (_bitmap4 != 0)
        count += _bitmap4->Count();

    if (_bitmap8 != 0)
        count += _bitmap8->Count();

    if (_bitmap16 != 0)
        count += _bitmap16->Count();

    if (_bitmap32 != 0)
        count += _bitmap32->Count();

    if (_bitmap64 != 0)
        count += _bitmap64->Count();

    return count;
}


/////////////////////////////////////////////////////////////////////
/// create a s-bitmap from an i-bitmap
///
/// @param iBitmap           the i-bitmap to copy
/////////////////////////////////////////////////////////////////////
void SeqBitmap::CreateSBitmap(const SeqBitmap &iBitmap)
{

#ifdef DEBUG_SEQBITMAPCOUNT
    _countCreateSBitmap++;
#endif

    if (iBitmap._bitmap4 != 0)
        _bitmap4->CreateSBitmap(*iBitmap._bitmap4);

    if (iBitmap._bitmap8 != 0)
        _bitmap8->CreateSBitmap(*iBitmap._bitmap8);

    if (iBitmap._bitmap16 != 0)
        _bitmap16->CreateSBitmap(*iBitmap._bitmap16);

    if (iBitmap._bitmap32 != 0)
        _bitmap32->CreateSBitmap(*iBitmap._bitmap32);

    if (iBitmap._bitmap64 != 0)
        _bitmap64->CreateSBitmap(*iBitmap._bitmap64);

}


/** @} */
