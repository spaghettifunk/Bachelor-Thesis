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

int const SeqBitmap::_lookupTableSize = 0x100;
int* SeqBitmap::_numOnesLookupTable = 0;
int* SeqBitmap::_compress4LookupTable = 0;
unsigned char** SeqBitmap::_compress8Table = 0;

// Variables used to gather usage statistics
int SeqBitmap::_countOr = 0;
int SeqBitmap::_countAnd = 0;
int SeqBitmap::_countCount = 0;
int SeqBitmap::_countCountZeros = 0;
int SeqBitmap::_countCountSmaller = 0;
int SeqBitmap::_countCreateSBitmap = 0;
int SeqBitmap::_countCreateCBitmap = 0;

unsigned int* SeqBitmap:: _memory4 = 0;
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


    // -- lookup table for compression
    int bitLookUp[8] = {128, 64, 32, 16, 8, 4, 2, 1};
    int i, j, k;

    _numOnesLookupTable = new int[_lookupTableSize];
    // loop over all 256 entries in the table
    for (i = 0; i < _lookupTableSize; i++)
    {

        // bit loops over the 8 different bits
        _numOnesLookupTable[i] = 0;
        for (j = 0; j < 8; j++)
        {
            if (i & bitLookUp[j])
                _numOnesLookupTable[i]++;
        }
    }


    // If compress4LookupTable = 0 -> all bits in char equal to 0
    //                           1 -> first four bits equal to 0
    //                           2 -> second four bits
    //                           3 -> loop over all 256 entries in the table
    
    _compress4LookupTable = new int[_lookupTableSize];
    _compress4LookupTable[0] = 0;

    for (i = 1; i < _lookupTableSize; i++)
    {

        // Or the first 4 bits together;
        // if the result is 0 then all 4 bits must be 0
        if ( (i&0xf) == 0)
            _compress4LookupTable[i] = 1;
        else if ( (i&0xf0) == 0 )
            // Or the second 4 bits together;
            // if the result is 0 then all 4 bits must be 0
            _compress4LookupTable[i] = 2;
        else
            _compress4LookupTable[i] = 3;

    }


    // compress8Table
    _compress8Table = new unsigned char * [_lookupTableSize];

    unsigned char value;
    int temp;
    int pos;
    for (i = 0; i < _lookupTableSize; i++)
    {
        _compress8Table[i] = new unsigned char[_lookupTableSize];
        for (j = 0; j < _lookupTableSize; j++)
        {
            value = 0;
            pos = 0;

            for (k = 0; k < 8; k++)
            {
                if (i & bitLookUp[k])
                {
                    temp = (bitLookUp[k] & j) << k;
                    value |= temp >> pos;
                    pos++;
                }
            }


            _compress8Table[i][j] = value;
        }

    }
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
    delete [] _memory4;
    delete [] _memory8;
    delete [] _memory16;
    delete [] _memory32;
    delete [] _memory64;
}



void SeqBitmap::Destroy()
{
    Bitmap4::Destroy();
    Bitmap8::Destroy();
    Bitmap16::Destroy();
    Bitmap32::Destroy();
    Bitmap64::Destroy();

    if (_numOnesLookupTable != 0)
        delete [] _numOnesLookupTable;

    if (_compress4LookupTable != 0)
        delete [] _compress4LookupTable;

    if (_compress8Table != 0)
    {
        for (int i = 0; i < _lookupTableSize; i++)
            if (_compress8Table[i] != 0)
                delete [] _compress8Table[i];
        delete [] _compress8Table;
    }
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

/////////////////////////////////////////////////////////////////////
/// create a cbitmap from an i-bitmap
///
/// @param iBitmap           the bitmap from which we create s-bitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::CreateCBitmap(const SeqBitmap &iBitmap)
{
#ifdef DEBUG_SEQBITMAPCOUNT
    _countCreateCBitmap++;
#endif

    if (iBitmap._bitmap4 != 0)
        _bitmap4->CreateCBitmap(*iBitmap._bitmap4);

    if (iBitmap._bitmap8 != 0)
        _bitmap8->CreateCBitmap(*iBitmap._bitmap8);

    if (iBitmap._bitmap16 != 0)
        _bitmap16->CreateCBitmap(*iBitmap._bitmap16);

    if (iBitmap._bitmap32 != 0)
        _bitmap32->CreateCBitmap(*iBitmap._bitmap32);

    if (iBitmap._bitmap64 != 0)
        _bitmap64->CreateCBitmap(*iBitmap._bitmap64);

}


/////////////////////////////////////////////////////////////////////
/// Print bitmap to an output stream
///
/// @param testFile           output file stream
/////////////////////////////////////////////////////////////////////
void SeqBitmap::PrintBitmap(ofstream &testFile)
{
    char buf[256];
    if (_bitmap4 != 0)
    {
        sprintf(buf, " 4 bits  ");
        testFile << buf << " " << _bitmap4->_memSizeInt << " : ";

        for (int i = 0; i < _bitmap4->_memSizeInt; i++)
        {
            sprintf(buf, "%x  ", _bitmap4->_memory[i]);
            testFile << buf;
        }
        testFile << endl;
        testFile << _bitmap4->Count() << endl;
    }

    if (_bitmap8 != 0)
    {
        sprintf(buf, " 8 bits  ");
        testFile << buf << " " << _bitmap8->_memSizeInt << " : ";

        for (int i = 0; i < _bitmap8->_memSizeInt; i++)
        {
            sprintf(buf, "%x  ", _bitmap8->_memory[i]);
            testFile << buf;
        }
        testFile << endl;
        testFile << _bitmap8->Count() << endl;
    }

    if (_bitmap16 != 0)
    {
        sprintf(buf, "16 bits  ");
        testFile << buf << " " << _bitmap16->_memSizeShort << " : ";

        for (int i = 0; i < _bitmap16->_memSizeShort; i++)
        {
            sprintf(buf, "%x  ", _bitmap16->_memory[i]);
            testFile << buf;
        }
        testFile << endl;
        testFile << _bitmap16->Count() << endl;
    }

    if (_bitmap32 != 0)
    {
        sprintf(buf, "32 bits  ");
        testFile << buf << " " << _bitmap32->_memSizeInt << " : ";

        for (int i = 0; i < _bitmap32->_memSizeInt; i++)
        {
            sprintf(buf, "%x  ", _bitmap32->_memory[i]);
            testFile << buf;
        }
        testFile << endl;
    }

    if (_bitmap64 != 0)
    {
        sprintf(buf, "64 bits  ");
        testFile << buf << " " << _bitmap64->_memSizeInt << " : ";
        for (int i = 0; i < _bitmap64->_memSizeInt; i++)
        {
            sprintf(buf, "%x  ", _bitmap64->_memory[i]);
            testFile << buf;
        }
        testFile << endl;
    }

}


/////////////////////////////////////////////////////////////////////
/// Returns the sizes to allocate for the new compressed f1 bitmap.
/// This will only be called by an OR bitmap during compression
/////////////////////////////////////////////////////////////////////
void SeqBitmap::CountSmaller(
    int &size4,
    int &size8,
    int &size16,
    int &size32,
    int &size64,
    ostream& debugFile)
{

#ifdef DEBUG_SEQBITMAPCOUNT
    _countCountSmaller++;
#endif

    // debugFile << "CountSmaller is called " << endl;
    // debugFile << "\tBitmap4 "<< endl;

    if (_bitmap4 != 0)
    {
        //  debugFile << "\t\tBitmap 4 is not empty" << endl;
        size4 = _bitmap4->Count();
    }
    else
    {
        //  debugFile << "\t\tBitmap 4 is empty" << endl;
        size4 = 0;
    }
    if (size4 % 2 != 0)
        size4++;

    // debugFile << "\t\t" << size4 <<endl;

    size8 = 0;
    size16 = 0;
    size32 = 0;
    size64 = 0;

    int i, count;
    unsigned char * memChar;

    // count in bitmap8

    // debugFile << "\tBitmap4 "<< endl;
    if (_bitmap8 != 0)
    {
        //  debugFile << "\t\tBitmap 8 is not empty" << endl;
        memChar = reinterpret_cast<unsigned char *>(_bitmap8->_memory);

        //  debugFile << "\t\t\t" << _bitmap8->_memSizeInt * 4 << endl;

        for (i = 0; i < _bitmap8->_memSizeInt*4; i++)
        {

            // debugFile << "\t\t\t\t" << memChar[i] << " ";
            // debugFile << "\t\t\t\t" << _numOnesLookupTable[memChar[i]] << endl;

            count = _numOnesLookupTable[memChar[i]];
            if (count > 4)
                size8++;
            else if (count > 0)
                size4++;
        }
        if (size4 % 2 != 0)
            size4++;
    }
    // else
    //  debugFile << "\t\tBitmap 8 is empty" << endl;

    // count in bitmap16
    if (_bitmap16 != 0)
    {
        //  debugFile << "\t\tBitmap 16 is not empty" << endl;
        memChar = reinterpret_cast<unsigned char *>(_bitmap16->_memory);
        for (i = 0; i < _bitmap16->_memSizeShort*2; i += 2)
        {
            count = _numOnesLookupTable[memChar[i]];
            count += _numOnesLookupTable[memChar[i + 1]];
            if (count > 8)
                size16++;
            else if (count > 4)
                size8++;
            else if (count > 0)
                size4++;
        }
        if (size4 % 2 != 0)
            size4++;
    }
    // else
    //  debugFile << "\t\tBitmap 16 is empty" << endl;

    // count in bitmap32
    if (_bitmap32 != 0)
    {
        //  debugFile << "\t\tBitmap 32 is not empty" << endl;
        memChar = reinterpret_cast<unsigned char *>(_bitmap32->_memory);
        for (i = 0; i < _bitmap32->_memSizeInt*4; i += 4)
        {
            count = _numOnesLookupTable[memChar[i]];
            count += _numOnesLookupTable[memChar[i + 1]];
            count += _numOnesLookupTable[memChar[i + 2]];
            count += _numOnesLookupTable[memChar[i + 3]];

            if (count > 16)
                size32++;
            else if (count > 8)
                size16++;
            else if (count > 4)
                size8++;
            else if (count > 0)
                size4++;
        }
        if (size4 % 2 != 0)
            size4++;
    }
    // else
    //  debugFile << "\t\tBitmap 32 is  empty" << endl;

    // count in bitmap64
    if (_bitmap64 != 0)
    {
        //  debugFile << "\t\tBitmap 64 is not empty" << endl;
        memChar = reinterpret_cast<unsigned char *>(_bitmap64->_memory);
        for (i = 0; i < _bitmap64->_memSizeInt * 4; i += 8)
        {
            count = _numOnesLookupTable[memChar[i]];
            count += _numOnesLookupTable[memChar[i + 1]];
            count += _numOnesLookupTable[memChar[i + 2]];
            count += _numOnesLookupTable[memChar[i + 3]];
            count += _numOnesLookupTable[memChar[i + 4]];
            count += _numOnesLookupTable[memChar[i + 5]];
            count += _numOnesLookupTable[memChar[i + 6]];
            count += _numOnesLookupTable[memChar[i + 7]];
            if (count > 32)
                size64++;
            else if (count > 16)
                size32++;
            else if (count > 8)
                size16++;
            else if (count > 4)
                size8++;
            else if (count > 0)
                size4++;
        }
        if (size4 % 2 != 0)
            size4++;
    }
    // else
    //  debugFile << "\t\tBitmap 64 is empty" << endl;

    // debugFile << "\tsize4 " << size4<<endl;
    // debugFile << "\tsize8 " << size8<<endl;
    // debugFile << "\tsize16 " << size16<<endl;
    // debugFile << "\tsize32 " << size32<<endl;
    // debugFile << "\tsize64 " << size64<<endl;
}

/////////////////////////////////////////////////////////////////////
/// Compress the Bitmap4 component of a SeqBitmap to a compressed SeqBitmap
///
/// @param refBitmap            The bitmap specifying which bits can be
///                                 compressed
/// @param compBitmap           The bitmap to compress
/// @param pos4                 The end position of the Bitmap4 in the
///                                 compressed SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Compress4(
    SeqBitmap *refBitmap,
    SeqBitmap *compBitmap,
    int &pos4)
{

    // we don't have to compress it
    if (_bitmap4 == 0)
        return ;

    int i;

    unsigned char copy = 0;
    bool upper4Bits = true;
    pos4 = 0;

    unsigned char* refMemChar =
        reinterpret_cast<unsigned char*>(refBitmap->_bitmap4->_memory);
    unsigned char* compMemChar =
        reinterpret_cast<unsigned char*>(compBitmap->_bitmap4->_memory);
    unsigned char* resultMemChar =
        reinterpret_cast<unsigned char*>(_bitmap4->_memory);

    // go over each character
    for (i = 0; i < refBitmap->_bitmap4->_memSizeInt*4; i++)
    {
        // check to see which of the 4 bits chunks have ones
        switch (_compress4LookupTable[refMemChar[i]])
        {
        case 1:
            // only the first four bits have ones
            if (upper4Bits)
            {
                copy = (compMemChar[i] & 0xf0);
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (compMemChar[i] >> 4);
                resultMemChar[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }
            break;
        case 2:
            // only the second four bits have ones
            if (upper4Bits)
            {
                copy = (compMemChar[i] << 4);
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (compMemChar[i] & 0xf);
                resultMemChar[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }
            break;
        case 3:
            resultMemChar[pos4] = compMemChar[i];
            pos4++;
            break;
        default:
            break;
        }
    }

    // copy one remaining set of 4 bits from "copy"
    if (!upper4Bits)
    {
        resultMemChar[pos4] = copy;
        pos4++;
    }
}


/////////////////////////////////////////////////////////////////////
/// Compress the Bitmap8 component of a SeqBitmap to a compressed SeqBitmap
///
/// @param refBitmap            The bitmap specifying which bits can be
///                                 compressed
/// @param compBitmap           The bitmap to compress
/// @param pos4                 The end position of the Bitmap4 in the
///                                 compressed SeqBitmap
/// @param pos8                 The end position of the Bitmap8 in the
///                                 compressed SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Compress8(
    SeqBitmap *refBitmap,
    SeqBitmap *compBitmap,
    int &pos4,
    int &pos8)
{

    // this bitmap will be gone
    if (_bitmap4 == 0 && _bitmap8 == 0)
        return ;

    unsigned char* refMemChar8 =
        reinterpret_cast<unsigned char*>(refBitmap->_bitmap8->_memory);
    unsigned char* compMemChar8 =
        reinterpret_cast<unsigned char*>(compBitmap->_bitmap8->_memory);
    unsigned char* resultMemChar4 = 0;
    unsigned char* resultMemChar8 = 0;
    pos8 = 0;

    if (_bitmap4 != 0)
        resultMemChar4 = reinterpret_cast<unsigned char*>(_bitmap4->_memory);

    if (_bitmap8 != 0)
        resultMemChar8 = reinterpret_cast<unsigned char*>(_bitmap8->_memory);



    // When an 8->4 compression is performed,
    // can only add the 4 bits to _bitmap4 when a group
    // of two 4 bit chunks exists
    unsigned char copy = 0;
    bool upper4Bits = true;
    int i, numOnes;

    for (i = 0; i < refBitmap->_bitmap8->_memSizeInt*4; i++)
    {
        numOnes = _numOnesLookupTable[refMemChar8[i]];

        if (numOnes > 4)
        {
            // Copy the current character into the current position of memChar8
            resultMemChar8[pos8] = compMemChar8[i];
            pos8++;
        }
        else if (numOnes > 0)
        {
            // numOnes is between 1 and 4, need to compress to 4
            // The compressed 4 bit chunk is the current f1 slice passed
            // into the generated lookup table.
            // Create the compressed 4 bit chunk, copy it into either the
            // lower or upper half of copy (depending on the current state
            // of Lower)



            if (upper4Bits)
            {
                copy = (_compress8Table[refMemChar8[i]][compMemChar8[i]]
                        & 0xf0);
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (_compress8Table[refMemChar8[i]][compMemChar8[i]]
                         >> 4);
                resultMemChar4[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }
        }
    }

    // copy one remaining set of 4 bits from "copy"
    if (!upper4Bits)
    {
        resultMemChar4[pos4] = copy;
        pos4++;
    }
}


/////////////////////////////////////////////////////////////////////
/// Compress the Bitmap16 component of a SeqBitmap to a compressed SeqBitmap
///
/// @param refBitmap            The bitmap specifying which bits can be
///                                 compressed
/// @param compBitmap           The bitmap to compress
/// @param pos4                 The end position of the Bitmap4 in the
///                                 compressed SeqBitmap
/// @param pos8                 The end position of the Bitmap8 in the
///                                 compressed SeqBitmap
/// @param pos16                The end position of the Bitmap16 in the
///                                 compressed SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Compress16(
    SeqBitmap *refBitmap,
    SeqBitmap *compBitmap,
    int &pos4,
    int &pos8,
    int &pos16)
{


    if (_bitmap4 == 0 && _bitmap8 == 0 && _bitmap16 == 0)
        return ;

    pos16 = 0;


    unsigned char* refMemChar16 =
        reinterpret_cast<unsigned char*>(refBitmap->_bitmap16->_memory);
    unsigned char* compMemChar16 =
        reinterpret_cast<unsigned char*>(compBitmap->_bitmap16->_memory);
    unsigned char* resultMemChar4 = 0;
    unsigned char* resultMemChar8 = 0;
    unsigned char* resultMemChar16 = 0;

    if (_bitmap4 != 0)
        resultMemChar4 = reinterpret_cast<unsigned char*>(_bitmap4->_memory);

    if (_bitmap8 != 0)
        resultMemChar8 = reinterpret_cast<unsigned char*>(_bitmap8->_memory);

    if (_bitmap16 != 0)
        resultMemChar16 = reinterpret_cast<unsigned char*>(_bitmap16->_memory);

    // When an 8->4 compression is performed,
    // can only add the 4 bits to _bitmap4 when a group of two 4 bit
    // chunks exists
    int i, numOnesChar1, numOnesChar2, numOnes;
    unsigned char copy = 0;
    bool upper4Bits = true;


    // Go through every other character
    // 16 is an array of shorts instead of int
    for (i = 0; i < refBitmap->_bitmap16->_memSizeShort*2; i += 2)
    {
        numOnesChar1 = _numOnesLookupTable[refMemChar16[i + 1]];
        numOnesChar2 = _numOnesLookupTable[refMemChar16[i]];
        numOnes = numOnesChar1 + numOnesChar2;

        if (numOnes > 8)
        {
            // Copy over the slice of 16 bits to the new _bitmap16
            resultMemChar16[pos16 + 1] = compMemChar16[i + 1];
            resultMemChar16[pos16] = compMemChar16[i];
            pos16 += 2;
        }
        else if (numOnes > 4)
        {
            // Compress 16 bits down to 8 bits
            // First, compress both chars in the 16 bit slice

            unsigned char slice1 = _compress8Table[refMemChar16[i + 1]]
                                   [compMemChar16[i + 1]];
            unsigned char slice2 = _compress8Table[refMemChar16[i]]
                                   [compMemChar16[i]];

            slice1 |= (slice2 >> numOnesChar1);

            resultMemChar8[pos8] = slice1;
            pos8++;
        }
        else if (numOnes > 0)
        {
            // numOnes is between 1 and 4
            // Compress 16 bits down to 4 bits
            // First, compress both chars in the 16 bit slice

            unsigned char slice1 = _compress8Table[refMemChar16[i + 1]]
                                   [compMemChar16[i + 1]];
            unsigned char slice2 = _compress8Table[refMemChar16[i]]
                                   [compMemChar16[i]];

            slice1 |= (slice2 >> numOnesChar1);

            // At this point slice1 must only take up the first four
            // bit positions in the char
            if (upper4Bits)
            {
                copy = (slice1 & 0xf0);
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (slice1 >> 4);
                resultMemChar4[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }
        }
    }

    // copy one remaining set of 4 bits from "copy"
    if (!upper4Bits)
    {
        resultMemChar4[pos4] = copy;
        pos4++;
    }
}


/////////////////////////////////////////////////////////////////////
/// Compress the Bitmap32 component of a SeqBitmap to a compressed SeqBitmap
///
/// @param refBitmap            The bitmap specifying which bits can be
///                                 compressed
/// @param compBitmap           The bitmap to compress
/// @param pos4                 The end position of the Bitmap4 in the
///                                 compressed SeqBitmap
/// @param pos8                 The end position of the Bitmap8 in the
///                                 compressed SeqBitmap
/// @param pos16                 The end position of the Bitmap16 in the
///                                 compressed SeqBitmap
/// @param pos32                 The end position of the Bitmap32 in the
///                                 compressed SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Compress32(
    SeqBitmap *refBitmap,
    SeqBitmap *compBitmap,
    int &pos4,
    int &pos8,
    int &pos16,
    int &pos32)
{

    if (_bitmap4 == 0 && _bitmap8 == 0 && _bitmap16 == 0 && _bitmap32 == 0)
        return ;


    unsigned char* refMemChar32 =
        reinterpret_cast<unsigned char*>(refBitmap->_bitmap32->_memory);
    unsigned char* compMemChar32 =
        reinterpret_cast<unsigned char*>(compBitmap->_bitmap32->_memory);
    unsigned char* resultMemChar4 = 0;
    unsigned char* resultMemChar8 = 0;
    unsigned char* resultMemChar16 = 0;
    unsigned char* resultMemChar32 = 0;

    if (_bitmap4 != 0)
        resultMemChar4 = reinterpret_cast<unsigned char*>(_bitmap4->_memory);

    if (_bitmap8 != 0)
        resultMemChar8 = reinterpret_cast<unsigned char*>(_bitmap8->_memory);

    if (_bitmap16 != 0)
        resultMemChar16 = reinterpret_cast<unsigned char*>(_bitmap16->_memory);

    if (_bitmap32 != 0)
        resultMemChar32 = reinterpret_cast<unsigned char*>(_bitmap32->_memory);


    // When an 8->4 compression is performed, can only add the 4 bits to
    // _bitmap4 when a group of two 4 bit chunks exists
    int i, numOnesChar1, numOnesChar2, numOnesChar3, numOnesChar4, numOnes;
    unsigned char copy = 0;
    bool upper4Bits = true;

    pos32 = 0;

    // Go through every 4th character
    for (i = 0; i < refBitmap->_bitmap32->_memSizeInt*4; i += 4)
    {
        numOnesChar1 = _numOnesLookupTable[refMemChar32[i + 3]];
        numOnesChar2 = _numOnesLookupTable[refMemChar32[i + 2]];
        numOnesChar3 = _numOnesLookupTable[refMemChar32[i + 1]];
        numOnesChar4 = _numOnesLookupTable[refMemChar32[i]];

        numOnes = numOnesChar1 + numOnesChar2 + numOnesChar3 + numOnesChar4;

        if (numOnes > 16)
        {
            // Copy over the slice of 32 bits to the new _bitmap32
            resultMemChar32[pos32 + 3] = compMemChar32[i + 3];
            resultMemChar32[pos32 + 2] = compMemChar32[i + 2];
            resultMemChar32[pos32 + 1] = compMemChar32[i + 1];
            resultMemChar32[pos32] = compMemChar32[i];
            pos32 += 4;
        }
        else if (numOnes > 8)
        {
            // Compress 32 bits down to 16 bits
            // Create the 4 8-bit slices based on the lookup table
            unsigned short slice1 = (_compress8Table[refMemChar32[i + 3]]
                                     [compMemChar32[i + 3]] << 8);
            unsigned short slice2 = (_compress8Table[refMemChar32[i + 2]]
                                     [compMemChar32[i + 2]] << 8);
            unsigned short slice3 = (_compress8Table[refMemChar32[i + 1]]
                                     [compMemChar32[i + 1]] << 8);
            unsigned short slice4 = (_compress8Table[refMemChar32[i]]
                                     [compMemChar32[i]] << 8);

            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // Put slice 1 and slice 2 into _bitmap16
            resultMemChar16[pos16 + 1] = (slice1 & 0xff00) >> 8;
            resultMemChar16[pos16] = (slice1 & 0xff);
            pos16 += 2;

        }
        else if (numOnes > 4)
        {
            // Compress 32 bits down to 8 bits
            // Create the 4 8-bit slices based on the lookup table
            unsigned char slice1 = _compress8Table[refMemChar32[i + 3]]
                                   [compMemChar32[i + 3]];
            unsigned char slice2 = _compress8Table[refMemChar32[i + 2]]
                                   [compMemChar32[i + 2]];
            unsigned char slice3 = _compress8Table[refMemChar32[i + 1]]
                                   [compMemChar32[i + 1]];
            unsigned char slice4 = _compress8Table[refMemChar32[i]]
                                   [compMemChar32[i]];

            // Shift these slices based on the # of ones
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);


            // Put slice 1 into _bitmap8
            resultMemChar8[pos8] = slice1;
            pos8++;
        }
        else if (numOnes > 0)
        {

            // numOnes is between 1 and 4
            // Compress 32 bits down to 4 bits
            // Create the 4 8-bit slices based on the lookup table
            unsigned char slice1 = _compress8Table[refMemChar32[i + 3]]
                                   [compMemChar32[i + 3]];
            unsigned char slice2 = _compress8Table[refMemChar32[i + 2]]
                                   [compMemChar32[i + 2]];
            unsigned char slice3 = _compress8Table[refMemChar32[i + 1]]
                                   [compMemChar32[i + 1]];
            unsigned char slice4 = _compress8Table[refMemChar32[i]]
                                   [compMemChar32[i]];

            // Shift these slices based on the # of ones
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // At this point slice1 must only take up the first four bit
            // positions in the char
            if (upper4Bits)
            {
                copy = (slice1 & 0xf0);
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (slice1 >> 4);
                resultMemChar4[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }
        }
    }

    // copy one remaining set of 4 bits from "copy"
    if (!upper4Bits)
    {
        resultMemChar4[pos4] = copy;
        pos4++;
    }
}

/////////////////////////////////////////////////////////////////////
/// Compress the Bitmap64 component of a SeqBitmap to a compressed SeqBitmap
///
/// @param refBitmap            The bitmap specifying which bits can be
///                                 compressed
/// @param compBitmap           The bitmap to compress
/// @param pos4                 The end position of the Bitmap4 in the
///                                 compressed SeqBitmap
/// @param pos8                 The end position of the Bitmap8 in the
///                                 compressed SeqBitmap
/// @param pos16                 The end position of the Bitmap16 in the
///                                 compressed SeqBitmap
/// @param pos32                 The end position of the Bitmap32 in the
///                                 compressed SeqBitmap
/// @param pos64                 The end position of the Bitmap64 in the
///                                 compressed SeqBitmap
/////////////////////////////////////////////////////////////////////
void SeqBitmap::Compress64(
    SeqBitmap *refBitmap,
    SeqBitmap *compBitmap,
    int &pos4,
    int &pos8,
    int &pos16,
    int &pos32,
    int &pos64)
{

    if (_bitmap4 == 0 &&
            _bitmap8 == 0 &&
            _bitmap16 == 0 &&
            _bitmap32 == 0 &&
            _bitmap64 == 0)
        return ;

    unsigned char* refMemChar64 =
        reinterpret_cast<unsigned char*>(refBitmap->_bitmap64->_memory);
    unsigned char* compMemChar64 =
        reinterpret_cast<unsigned char*>(compBitmap->_bitmap64->_memory);
    unsigned char* resultMemChar4 = 0;
    unsigned char* resultMemChar8 = 0;
    unsigned char* resultMemChar16 = 0;
    unsigned char* resultMemChar32 = 0;
    unsigned char* resultMemChar64 = 0;

    if (_bitmap4 != 0)
        resultMemChar4 = reinterpret_cast<unsigned char*>(_bitmap4->_memory);

    if (_bitmap8 != 0)
        resultMemChar8 = reinterpret_cast<unsigned char*>(_bitmap8->_memory);

    if (_bitmap16 != 0)
        resultMemChar16 = reinterpret_cast<unsigned char*>(_bitmap16->_memory);

    if (_bitmap32 != 0)
        resultMemChar32 = reinterpret_cast<unsigned char*>(_bitmap32->_memory);

    if (_bitmap64 != 0)
        resultMemChar64 = reinterpret_cast<unsigned char*>(_bitmap64->_memory);


    // When an 8->4 compression is performed, can only add the 4 bits
    // to _bitmap4 when a group of two 4 bit chunks exists
    int i, numOnesChar1, numOnesChar2, numOnesChar3, numOnesChar4,
    numOnesChar5, numOnesChar6, numOnesChar7, numOnesChar8, numOnes;
    unsigned char copy = 0;
    bool upper4Bits = true;

    pos64 = 0;


    // Go through every 8th character
    for (i = 0; i < refBitmap->_bitmap64->_memSizeInt*4; i += 8)
    {
        numOnesChar1 = _numOnesLookupTable[refMemChar64[i + 3]];
        numOnesChar2 = _numOnesLookupTable[refMemChar64[i + 2]];
        numOnesChar3 = _numOnesLookupTable[refMemChar64[i + 1]];
        numOnesChar4 = _numOnesLookupTable[refMemChar64[i]];
        numOnesChar5 = _numOnesLookupTable[refMemChar64[i + 7]];
        numOnesChar6 = _numOnesLookupTable[refMemChar64[i + 6]];
        numOnesChar7 = _numOnesLookupTable[refMemChar64[i + 5]];
        numOnesChar8 = _numOnesLookupTable[refMemChar64[i + 4]];

        numOnes = numOnesChar1 +
                  numOnesChar2 +
                  numOnesChar3 +
                  numOnesChar4 +
                  numOnesChar5 +
                  numOnesChar6 +
                  numOnesChar7 +
                  numOnesChar8;

        if (numOnes > 32)
        {
            // Copy over the slice of 64 bits to the new _bitmap32
            resultMemChar64[pos64 + 3] = compMemChar64[i + 3];
            resultMemChar64[pos64 + 2] = compMemChar64[i + 2];
            resultMemChar64[pos64 + 1] = compMemChar64[i + 1];
            resultMemChar64[pos64] = compMemChar64[i];
            resultMemChar64[pos64 + 7] = compMemChar64[i + 7];
            resultMemChar64[pos64 + 6] = compMemChar64[i + 6];
            resultMemChar64[pos64 + 5] = compMemChar64[i + 5];
            resultMemChar64[pos64 + 4] = compMemChar64[i + 4];
            pos64 += 8;
        }
        else if (numOnes > 16)
        {
            // Compress 64 bits down to 32 bits
            // Create the 8 8-bit slices based on the lookup table
            unsigned int slice1 = _compress8Table[refMemChar64[i + 3]]
                                  [compMemChar64[i + 3]] << 24;
            unsigned int slice2 = _compress8Table[refMemChar64[i + 2]]
                                  [compMemChar64[i + 2]] << 24;
            unsigned int slice3 = _compress8Table[refMemChar64[i + 1]]
                                  [compMemChar64[i + 1]] << 24;
            unsigned int slice4 = _compress8Table[refMemChar64[i]]
                                  [compMemChar64[i]] << 24;
            unsigned int slice5 = _compress8Table[refMemChar64[i + 7]]
                                  [compMemChar64[i + 7]] << 24;
            unsigned int slice6 = _compress8Table[refMemChar64[i + 6]]
                                  [compMemChar64[i + 6]] << 24;
            unsigned int slice7 = _compress8Table[refMemChar64[i + 5]]
                                  [compMemChar64[i + 5]] << 24;
            unsigned int slice8 = _compress8Table[refMemChar64[i + 4]]
                                  [compMemChar64[i + 4]] << 24;

            // Shift these slices based on the # of ones
            slice7 |= (slice8 >> numOnesChar7);
            slice6 |= (slice7 >> numOnesChar6);
            slice5 |= (slice6 >> numOnesChar5);
            slice4 |= (slice5 >> numOnesChar4);
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // Put slices 1-4 into _bitmap32
            resultMemChar32[pos32 + 3] = (slice1 & 0xff000000) >> 24;
            resultMemChar32[pos32 + 2] = (slice1 & 0xff0000) >> 16;
            resultMemChar32[pos32 + 1] = (slice1 & 0xff00) >> 8;
            resultMemChar32[pos32] = (slice1 & 0xff);
            pos32 += 4;
        }
        else if (numOnes > 8)
        {
            // Compress 64 bits down to 16 bits
            // Create the 8 8-bit slices based on the lookup table
            unsigned short slice1 = _compress8Table[refMemChar64[i + 3]]
                                    [compMemChar64[i + 3]] << 8;
            unsigned short slice2 = _compress8Table[refMemChar64[i + 2]]
                                    [compMemChar64[i + 2]] << 8;
            unsigned short slice3 = _compress8Table[refMemChar64[i + 1]]
                                    [compMemChar64[i + 1]] << 8;
            unsigned short slice4 = _compress8Table[refMemChar64[i]]
                                    [compMemChar64[i]] << 8;
            unsigned short slice5 = _compress8Table[refMemChar64[i + 7]]
                                    [compMemChar64[i + 7]] << 8;
            unsigned short slice6 = _compress8Table[refMemChar64[i + 6]]
                                    [compMemChar64[i + 6]] << 8;
            unsigned short slice7 = _compress8Table[refMemChar64[i + 5]]
                                    [compMemChar64[i + 5]] << 8;
            unsigned short slice8 = _compress8Table[refMemChar64[i + 4]]
                                    [compMemChar64[i + 4]] << 8;

            // Shift these slices based on the # of ones
            slice7 |= (slice8 >> numOnesChar7);
            slice6 |= (slice7 >> numOnesChar6);
            slice5 |= (slice6 >> numOnesChar5);
            slice4 |= (slice5 >> numOnesChar4);
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // Put slice 1 and slice 2 into _bitmap16
            resultMemChar16[pos16 + 1] = (slice1 & 0xff00) >> 8;
            resultMemChar16[pos16] = (slice1 & 0xff);
            pos16 += 2;
        }
        else if (numOnes > 4)
        {
            // Compress 64 bits down to 8 bits
            // Create the 8 8-bit slices based on the lookup table
            unsigned char slice1 = _compress8Table[refMemChar64[i + 3]]
                                   [compMemChar64[i + 3]];
            unsigned char slice2 = _compress8Table[refMemChar64[i + 2]]
                                   [compMemChar64[i + 2]];
            unsigned char slice3 = _compress8Table[refMemChar64[i + 1]]
                                   [compMemChar64[i + 1]];
            unsigned char slice4 = _compress8Table[refMemChar64[i]]
                                   [compMemChar64[i]];
            unsigned char slice5 = _compress8Table[refMemChar64[i + 7]]
                                   [compMemChar64[i + 7]];
            unsigned char slice6 = _compress8Table[refMemChar64[i + 6]]
                                   [compMemChar64[i + 6]];
            unsigned char slice7 = _compress8Table[refMemChar64[i + 5]]
                                   [compMemChar64[i + 5]];
            unsigned char slice8 = _compress8Table[refMemChar64[i + 4]]
                                   [compMemChar64[i + 4]];

            // Shift these slices based on the # of ones
            slice7 |= (slice8 >> numOnesChar7);
            slice6 |= (slice7 >> numOnesChar6);
            slice5 |= (slice6 >> numOnesChar5);
            slice4 |= (slice5 >> numOnesChar4);
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // Put slice 1 into _bitmap8
            resultMemChar8[pos8] = slice1;
            pos8++;
        }
        else if (numOnes > 0)
        {
            // numOnes is between 1 and 4
            // Compress 64 bits down to 4 bits
            // Create the 8 8-bit slices based on the lookup table
            unsigned char slice1 = _compress8Table[refMemChar64[i + 3]]
                                   [compMemChar64[i + 3]];
            unsigned char slice2 = _compress8Table[refMemChar64[i + 2]]
                                   [compMemChar64[i + 2]];
            unsigned char slice3 = _compress8Table[refMemChar64[i + 1]]
                                   [compMemChar64[i + 1]];
            unsigned char slice4 = _compress8Table[refMemChar64[i]]
                                   [compMemChar64[i]];
            unsigned char slice5 = _compress8Table[refMemChar64[i + 7]]
                                   [compMemChar64[i + 7]];
            unsigned char slice6 = _compress8Table[refMemChar64[i + 6]]
                                   [compMemChar64[i + 6]];
            unsigned char slice7 = _compress8Table[refMemChar64[i + 5]]
                                   [compMemChar64[i + 5]];
            unsigned char slice8 = _compress8Table[refMemChar64[i + 4]]
                                   [compMemChar64[i + 4]];

            // Shift these slices based on the # of ones
            slice7 |= (slice8 >> numOnesChar7);
            slice6 |= (slice7 >> numOnesChar6);
            slice5 |= (slice6 >> numOnesChar5);
            slice4 |= (slice5 >> numOnesChar4);
            slice3 |= (slice4 >> numOnesChar3);
            slice2 |= (slice3 >> numOnesChar2);
            slice1 |= (slice2 >> numOnesChar1);

            // At this point slice1 must only take up the first four bit
            // positions in the char
            if (upper4Bits)
            {
                copy = slice1;
                upper4Bits = false;
            }
            else
            {
                // now copy "copy" over to the output bitmap
                copy |= (slice1 >> 4);
                resultMemChar4[pos4] = copy;
                pos4++;
                upper4Bits = true;
            }

        }
    }


    // copy one remaining set of 4 bits from "copy"
    if (!upper4Bits)
    {
        resultMemChar4[pos4] = (copy & 0xf0);
        pos4++;
    }

}

void SeqBitmap::printSizes(ostream& out)
{
    int size4, size8, size16, size32, size64;
    if (_bitmap4)
    {
        size4 = _bitmap4->getIntSize() * 8;
    }
    else
    {
        size4 = 0;
    }
    if (_bitmap8)
    {
        size8 = _bitmap8->getIntSize() * 4;
    }
    else
    {
        size8 = 0;
    }
    if (_bitmap16)
    {
        size16 = _bitmap16->getIntSize() * 2;
    }
    else
    {
        size16 = 0;
    }
    if (_bitmap32)
    {
        size32 = _bitmap32->getIntSize() * 1;
    }
    else
    {
        size32 = 0;
    }
    if (_bitmap64)
    {
        size64 = _bitmap64->getIntSize() / 2;
    }
    else
    {
        size64 = 0;
    }

    out << size4 << " "
    << size8 << " "
    << size16 << " "
    << size32 << " "
    << size64 << endl;
}

/** @} */
