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
/// SeqBitmap.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __SEQBITMAP_H__
#define __SEQBITMAP_H__

#include "Bitmap4.h"
#include "Bitmap8.h"
#include "Bitmap16.h"
#include "Bitmap32.h"
#include "Bitmap64.h"
#include "Tables.h"
#include <fstream>

using namespace std;

/////////////////////////////////////////////////////////////////////
/// @addtogroup BitmapProcessingGroup Bitmap Processing
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A representation of a sequence (or an item)
/////////////////////////////////////////////////////////////////////
class SeqBitmap
{

    // we probably need access of Bitmap's private variables if we
    // put our compress code in this class
    friend class Bitmap4;
    friend class Bitmap8;
    friend class Bitmap16;
    friend class Bitmap32;
    friend class Bitmap64;

public:

    static void Init();
    static void MemAlloc(
        int size4,
        int size8,
        int size16,
        int size32,
        int size64);

    static void MemDealloc();
    static void Destroy();

    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for the Bitmap
    ///
    /// @param size4             size (number of 4-bits) of each bitmap
    /// @param size8             size (number of 4-bits) of each bitmap
    /// @param size16            size (number of 4-bits) of each bitmap
    /// @param size32            size (number of 4-bits) of each bitmap
    /// @param size64            size (number of 4-bits) of each bitmap
    /////////////////////////////////////////////////////////////////////
    SeqBitmap(int size4, int size8, int size16, int size32, int size64)
    {
        if (size4 > 0)
            _bitmap4 = new Bitmap4(size4, _memory4);
        else
            _bitmap4 = 0;

        if (size8 > 0)
            _bitmap8 = new Bitmap8(size8, _memory8);
        else
            _bitmap8 = 0;

        if (size16 > 0)
            _bitmap16 = new Bitmap16(size16, _memory16);
        else
            _bitmap16 = 0;

        if (size32 > 0)
            _bitmap32 = new Bitmap32(size32, _memory32);
        else
            _bitmap32 = 0;

        if (size64 > 0)
            _bitmap64 = new Bitmap64(size64, _memory64);
        else
            _bitmap64 = 0;



        //debugFile.open("seqPatDebug.txt");
    };

    /////////////////////////////////////////////////////////////////////
    /// Copy constructor
    ///
    /// @param b                 Bitmap to copy
    /////////////////////////////////////////////////////////////////////
    SeqBitmap(SeqBitmap &b)
    {
        // Copy information from source bitmap
        if (b._bitmap4 != 0)
            _bitmap4 = new Bitmap4(*b._bitmap4, _memory4);
        else
            _bitmap4 = 0;

        if (b._bitmap8 != 0)
            _bitmap8 = new Bitmap8(*b._bitmap8, _memory8);
        else
            _bitmap8 = 0;

        if (b._bitmap16 != 0)
            _bitmap16 = new Bitmap16(*b._bitmap16, _memory16);
        else
            _bitmap16 = 0;

        if (b._bitmap32 != 0)
            _bitmap32 = new Bitmap32(*b._bitmap32, _memory32);
        else
            _bitmap32 = 0;

        if (b._bitmap64 != 0)
            _bitmap64 = new Bitmap64(*b._bitmap64, _memory64);
        else
            _bitmap64 = 0;
    };

    /////////////////////////////////////////////////////////////////////
    /// Deallocate memory for the Bitmap
    /////////////////////////////////////////////////////////////////////
    ~SeqBitmap()
    {
        if (_bitmap4 != 0)
            delete _bitmap4;

        if (_bitmap8 != 0)
            delete _bitmap8;

        if (_bitmap16 != 0)
            delete _bitmap16;

        if (_bitmap32 != 0)
            delete _bitmap32;

        if (_bitmap64 != 0)
            delete _bitmap64;
    };

    /////////////////////////////////////////////////////////////////////
    /// Pop this SeqBitmap's memory from the global stack (allocation is
    ///     done in the constructor, but deallocation is not done in the
    ///     destructor
    /////////////////////////////////////////////////////////////////////
    void Deallocate()
    {
        if (_bitmap4 != 0)
            _bitmap4->Deallocate(_memory4);
        if (_bitmap8 != 0)
        {
            _bitmap8->Deallocate(_memory8);
        }
        if (_bitmap16 != 0)
        {
            _bitmap16->Deallocate(_memory16);
        }
        if (_bitmap32 != 0)
        {
            _bitmap32->Deallocate(_memory32);
        }
        if (_bitmap64 != 0)
        {
            _bitmap64->Deallocate(_memory64);
        }
    }

    /////////////////////////////////////////////////////////////////////
    /// Fill in a 1 in a certain position
    ///
    /// @param j                 bit position in Bitmap to be changed
    /// @param bitmapID          the bitmap we want to fill data in
    /////////////////////////////////////////////////////////////////////
    void FillEmptyPosition(int bitmapID, int j)
    {
        switch (bitmapID)
        {
        case 0:
            _bitmap4->FillEmptyPosition(j);
            break;
        case 1:
            _bitmap8->FillEmptyPosition(j);
            break;
        case 2:
            _bitmap16->FillEmptyPosition(j);
            break;
        case 3:
            _bitmap32->FillEmptyPosition(j);
            break;
        case 4:
            _bitmap64->FillEmptyPosition(j);
            break;
        }

    };

    /////////
    /// @return the aggregate memory size in # of ints
    /////////
    int memSize()
    {
        int totalSize = 0;
        if (_bitmap4 != 0)
            totalSize += _bitmap4->_memSizeInt;

        if (_bitmap8 != 0)
            totalSize += _bitmap8->_memSizeInt;

        if (_bitmap16 != 0)
            totalSize += _bitmap16->_memSizeInt;

        if (_bitmap32 != 0)
            totalSize += _bitmap32->_memSizeInt;

        if (_bitmap64 != 0)
            totalSize += _bitmap64->_memSizeInt;

        return totalSize * 32;
    };

    // --- other functions, refer to SeqBitmap.cpp for details
    void Or(const SeqBitmap &b1, const SeqBitmap &b2);
    void And(const SeqBitmap &b1, const SeqBitmap &b2);
    int Count();
    void CreateSBitmap(const SeqBitmap &iBitmap);
    void CreateCBitmap(const SeqBitmap &iBitmap);

    void PrintBitmap(ofstream& testFile);

    void CountSmaller(
        int &size4,
        int &size8,
        int &size16,
        int &size32,
        int &size64,
        ostream& out);

    void Compress4(
        SeqBitmap *refBitmap,
        SeqBitmap *compBitmap,
        int &pos4);
    void Compress8(
        SeqBitmap *refBitmap,
        SeqBitmap *compBitmap,
        int &pos4,
        int &pos8);
    void Compress16(
        SeqBitmap *refBitmap,
        SeqBitmap *compBitmap,
        int &pos4,
        int &pos8,
        int &pos16);
    void Compress32(
        SeqBitmap *refBitmap,
        SeqBitmap *compBitmap,
        int &pos4,
        int &pos8,
        int &pos16,
        int &pos32);
    void Compress64(
        SeqBitmap *refBitmap,
        SeqBitmap *compBitmap,
        int &pos4,
        int &pos8,
        int &pos16,
        int &pos32,
        int &pos64);

    void printSizes(ostream& out);

    static int _countOr;
    static int _countAnd;
    static int _countCount;
    static int _countCountZeros;
    static int _countCountSmaller;
    static int _countCreateSBitmap;
    static int _countCreateCBitmap;

    // Pointers to memory buffers for bitmaps
    static unsigned int* _memory4;
    static unsigned int* _memory8;
    static unsigned short* _memory16;
    static unsigned int* _memory32;
    static unsigned int* _memory64;

    // lookup table for counting
    static int const _lookupTableSize;
    static int* _numOnesLookupTable;
    static int* _compress4LookupTable;
    static unsigned char** _compress8Table;

    // Global sizes of the bitmap stacks
    static int _size4;
    static int _size8;
    static int _size16;
    static int _size32;
    static int _size64;

    Bitmap4* _bitmap4;
    Bitmap8* _bitmap8;
    Bitmap16* _bitmap16;
    Bitmap32* _bitmap32;
    Bitmap64* _bitmap64;
};

/** @} */

#endif
