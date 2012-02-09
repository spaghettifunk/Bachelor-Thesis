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
/// DatasetInfo.h
///
/////////////////////////////////////////////////////////////////////

#ifndef __DATASETINFO_H
#define __DATASETINFO_H

/////////////////////////////////////////////////////////////////////
/// @addtogroup FileInputGroup
/** @{ */

#include "SeqBitmap.h"

typedef struct
{
    int count;
    int index;
}
CountInfo;

/////////////////////////////////////////////////////////////////////
/// A class for representing the info gathered from the dataset
/////////////////////////////////////////////////////////////////////
class DatasetInfo
{
public:

    // -- constructor
    DatasetInfo()
    {
        custCount = 0;
        minSup = 0;
        f1Size = 0;
        f1BufSize = 0;
        maxCustTrans = 0;
        f1Buff = 0;
        f1NameBuff = 0;
        sListBuff = 0;
        iListBuff = 0;
        countBuff = 0;

    }

    // -- destructor
    ~DatasetInfo()
    {
        if (f1Buff != 0)
        {
            for (int i = 0; i < f1Size; i++)
                delete f1Buff[i];
            delete [] f1Buff;
        }

        if (f1NameBuff != 0)
            delete [] f1NameBuff;

        if (iListBuff != 0)
            delete [] iListBuff;

        if (sListBuff != 0)
            delete [] sListBuff;

        if (countBuff != 0)
            delete [] countBuff;

        if (bitmapSizes != 0)
            delete [] bitmapSizes;
    }


    // -- variables

    /// number of customers who have transactions in the data set
    int custCount;


    /// the minimum support (the minimum number of customers)
    int minSup;

    /// the number of frequent 1 itemsets
    int f1Size;

    /// the size of the f1Buff and f1NameBuff
    int f1BufSize;

    /// the maximum number of transactions of a customer
    int maxCustTrans;

    /// the buffer for frequent 1 itemsets (data are stored in the bitmaps)
    SeqBitmap** f1Buff;

    /// the buffer for bitmap4s
    SeqBitmap** bitmap4Buff;

    /// the buffer for the info of item names
    int* f1NameBuff;

    /// the buffer for the lists of s-steps
    int* sListBuff;

    /// the buffer for the lists of i-steps
    int* iListBuff;

    /// the buffer for the counts
    CountInfo *countBuff;

    /// size of each bitmap
    int* bitmapSizes;

};

/** @} */

#endif
