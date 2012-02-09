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
/// Spam.cpp
///
/////////////////////////////////////////////////////////////////////

// Toggle whether the output file format should be the same as that of
// PrefixSpan (for Perlscript regression test purposes)
//#define ___PREFIXSPAN___

// Toggle whether at each step, the frequent i-step and s-step extensions
// should be traversed in order of decreasing support
//#define ___REORDERING___

// Toggle whether bitmap compression will be used. If not, all bitmaps
// encapsulated within SeqBitmap will be Bitmap32s
//#define __COMPRESSION__

// Toggle whether various debugging statistics will be included with the
// output file
//#define __STATS__

// Toggle whether or not to compress the bitmaps at every single level
// (when compression is turned on)
//#define __ALLCOMPRESS__


#include "StringMap.h"
#include "SeqBitmap.h"
#include "TreeNode.h"
#include "DatasetInfo.h"
#include "Stats.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;

/// @defgroup GlobalVariables Global Variables
/// Global variables
/** @{ */

// Support count of the current node
CountInfo* TreeNode::countList = 0;

ofstream summaryFile;

// prototypes
int compare( const void *arg1, const void *arg2 );
void LogSequence(const int c);
void FindSequentialPatterns(TreeNode* curNode);
void StartMining(DatasetInfo* info);
void PrintError();
void CreateOrBitmap(
                    SeqBitmap** f1,
                    int* indexList,
                    int indexLength,
                    SeqBitmap*& orBitmap);
void Compress(
              SeqBitmap* refBitmap,
              SeqBitmap* tempAndBitmap,
              SeqBitmap*& returnBitmap,
              SeqBitmap** f1,
              SeqBitmap** newF1,
              int* indexList,
              int indexLength);


// Variables for tree nodes
int minSup;             ///< min sup as a transaction count
TreeNode** nodeBuff;    ///< node buffer
int* tempIndexList;     ///< stored the result of combining i-list and s-list
bool* indexExists;      ///< for combining i-list and s-list

// Variables for printing output
ofstream outFile;
int **sequentialPatterns;
int *elementSize;
int sequenceLength;
bool outputSeq;
bool stdoutSpecified;

StringMap *custStrMap;// StringMap objects that map the IDs into their string names
StringMap *transStrMap;
StringMap *itemStrMap;
bool isStringFile;      // Whether or not we should print the output as strings (because the input was strings)


// Misc global variables

/// no compression will be done if the bitmap size (_sizeShort)
/// is small than this
int minCompSize;

/// the ratio at which we compress the bitmaps
double emptySpaceRatio;

/// Used to test whether compression should be done at every level
int compLevel;

int totalCust;
int testCount = 0;
int numCompress = 0;

// Jay: Temporary global for compression debugging
int globalLevel;
#ifdef __STATS__
Stats *statistics;
#endif
/** @} */


/// @defgroup GlobalFunctions Global Functions
/// Global functions
/** @{ */

// function for reading input, refer to FileInput.cpp for details
DatasetInfo* ReadDataset(
    bool isBinaryFile,
    bool isStringFile,
    char *filename,
    double minSupPercent,
    StringMap *& custStrMap,
    StringMap *& transStrMap,
    StringMap *& itemStrMap);

int compare( const void *arg1, const void *arg2 )
{
    // Compare all of both strings:
    return ( ((CountInfo* ) arg1)->count - (( CountInfo* ) arg2)->count );
}

void LogStdoutSequence(const int c);
void LogFileSequence(const int c);

void LogSequence(const int c)
{
    if (stdoutSpecified)
        LogStdoutSequence(c);
    else
        LogFileSequence(c);
}


void LogStdoutSequence(const int c)
{
    // Output the sequence to the outFile

#ifdef ___PREFIXSPAN___

    char buf[256];
    // compare with prefix span, need to output the sequence in the same
    // way that PrefixSpan does
    for (int j = 0; j <= sequenceLength; j++)
    {

        cout << "(";

        //  assert(sequentialPatterns[j][0] < indexLength);

        //  cout << indexList[sequentialPatterns[j][0]];
        //  for (int m = 1; m <= elementSize[j]; m++) {
        //   assert(sequentialPatterns[j][m] < indexLength);
        //   cout << " " << indexList[sequentialPatterns[j][m]];

        if (isStringFile)
            cout << itemStrMap->GetValue(sequentialPatterns[j][0]);
        else
            cout << sequentialPatterns[j][0];

        for (int m = 1; m <= elementSize[j]; m++)
        {
            if (isStringFile)
                cout << " " << itemStrMap->GetValue(sequentialPatterns[j][m]);
            else
                cout << " " << sequentialPatterns[j][m];
        }
        cout << ") ";
    }

    sprintf(buf, "%.6f", (double)c / (double)totalCust);
    cout << ": " << buf << endl;

#else

    int m;
    // compare with spade, output the data in the same format that Spade does

    int* curItemset;
    int temp;

    if (elementSize[0] >= 1)
    {

        curItemset = new int[elementSize[0] + 1];

        for (m = 0; m <= elementSize[0]; m++)
            curItemset[m] = sequentialPatterns[0][m];


        for (int kk = 0; kk <= elementSize[0]; kk++)
        {
            for (int jj = kk + 1; jj <= elementSize[0]; jj++)
            {
                if (curItemset[kk] > curItemset[jj])
                {
                    temp = curItemset[kk];
                    curItemset[kk] = curItemset[jj];
                    curItemset[jj] = temp;
                }
            }
        }

        for (m = 0; m <= elementSize[0]; m++)
        {
            if (isStringFile)
                cout << itemStrMap->GetValue(curItemset[m]) << " ";
            else
                cout << curItemset[m] << " " ;
        }

        delete [] curItemset;

    }
    else
        if (isStringFile)
            cout << itemStrMap->GetValue(sequentialPatterns[0][0]) << " ";
        else
            cout << sequentialPatterns[0][0] << " " ;


    for (int j = 1; j <= sequenceLength; j++)
    {
        cout << "-1 ";


        if (elementSize[j] >= 1)
        {
            curItemset = new int[elementSize[j] + 1];

            for (m = 0; m <= elementSize[j]; m++)
                curItemset[m] = sequentialPatterns[j][m];

            for (int kk = 0; kk <= elementSize[j]; kk++)
                for (int jj = kk + 1; jj <= elementSize[j]; jj++)
                    if (curItemset[kk] > curItemset[jj])
                    {
                        temp = curItemset[kk];
                        curItemset[kk] = curItemset[jj];
                        curItemset[jj] = temp;
                    }

            if (isStringFile)
                cout << itemStrMap->GetValue(curItemset[0]) << " ";
            else
                cout << curItemset[0] << " " ;

            for (m = 1; m <= elementSize[j]; m++)
            {
                if (isStringFile)
                    cout << itemStrMap->GetValue(curItemset[m]) << " ";
                else
                    cout << curItemset[m] << " " ;
            }

            delete [] curItemset;
        }
        else
            if (isStringFile)
                cout << itemStrMap->GetValue(sequentialPatterns[j][0]);
            else
                cout << sequentialPatterns[j][0] << " " ;

    }
    cout << "- " << c << endl;


#endif

}



void LogFileSequence(const int c)
{
    // Output the sequence to the outFile

#ifdef ___PREFIXSPAN___

    char buf[256];
    // compare with prefix span, need to output the sequence in the same
    // way that PrefixSpan does
    for (int j = 0; j <= sequenceLength; j++)
    {

        outFile << "(";

        //  assert(sequentialPatterns[j][0] < indexLength);

        //  outFile << indexList[sequentialPatterns[j][0]];
        //  for (int m = 1; m <= elementSize[j]; m++) {
        //   assert(sequentialPatterns[j][m] < indexLength);
        //   outFile << " " << indexList[sequentialPatterns[j][m]];

        if (isStringFile)
            outFile << itemStrMap->GetValue(sequentialPatterns[j][0]);
        else
            outFile << sequentialPatterns[j][0];

        for (int m = 1; m <= elementSize[j]; m++)
        {
            if (isStringFile)
                outFile << " " << itemStrMap->GetValue(sequentialPatterns[j][m]);
            else
                outFile << " " << sequentialPatterns[j][m];
        }
        outFile << ") ";
    }

    sprintf(buf, "%.6f", (double)c / (double)totalCust);
    outFile << ": " << buf << endl;

#else

    int m;
    // compare with spade, output the data in the same format that Spade does

    int* curItemset;
    int temp;

    if (elementSize[0] >= 1)
    {

        curItemset = new int[elementSize[0] + 1];

        for (m = 0; m <= elementSize[0]; m++)
            curItemset[m] = sequentialPatterns[0][m];


        for (int kk = 0; kk <= elementSize[0]; kk++)
        {
            for (int jj = kk + 1; jj <= elementSize[0]; jj++)
            {
                if (curItemset[kk] > curItemset[jj])
                {
                    temp = curItemset[kk];
                    curItemset[kk] = curItemset[jj];
                    curItemset[jj] = temp;
                }
            }
        }

        for (m = 0; m <= elementSize[0]; m++)
        {
            if (isStringFile)
                outFile << itemStrMap->GetValue(curItemset[m]) << " ";
            else
                outFile << curItemset[m] << " " ;
        }

        delete [] curItemset;

    }
    else
        if (isStringFile)
            outFile << itemStrMap->GetValue(sequentialPatterns[0][0]) << " ";
        else
            outFile << sequentialPatterns[0][0] << " " ;


    for (int j = 1; j <= sequenceLength; j++)
    {
        outFile << "-1 ";


        if (elementSize[j] >= 1)
        {
            curItemset = new int[elementSize[j] + 1];

            for (m = 0; m <= elementSize[j]; m++)
                curItemset[m] = sequentialPatterns[j][m];

            for (int kk = 0; kk <= elementSize[j]; kk++)
                for (int jj = kk + 1; jj <= elementSize[j]; jj++)
                    if (curItemset[kk] > curItemset[jj])
                    {
                        temp = curItemset[kk];
                        curItemset[kk] = curItemset[jj];
                        curItemset[jj] = temp;
                    }

            if (isStringFile)
                outFile << itemStrMap->GetValue(curItemset[0]) << " ";
            else
                outFile << curItemset[0] << " " ;

            for (m = 1; m <= elementSize[j]; m++)
            {
                if (isStringFile)
                    outFile << itemStrMap->GetValue(curItemset[m]) << " ";
                else
                    outFile << curItemset[m] << " " ;
            }

            delete [] curItemset;
        }
        else
            if (isStringFile)
                outFile << itemStrMap->GetValue(sequentialPatterns[j][0]);
            else
                outFile << sequentialPatterns[j][0] << " " ;

    }
    outFile << "- " << c << endl;


#endif

}



///////////////////////////////////////////////////////
/// OR's all of the frequent-1 itemset bitmaps together
/// This is used to create the refBitmap for bitmap compression
///
/// @param f1                   The frequent-1 itemset bitmaps
/// @param indexList            The list of item names for the 1 itemsets
///                                 that are still frequent
/// @param indexLength          The length of indexList
///
/// @param orBitmap             [output]The OR'd bitmap. This tells us which
///                                 bits can be used in operations further
///                                 down this branch of the sequence tree
///////////////////////////////////////////////////////
void CreateOrBitmap(
    SeqBitmap** f1,
    int* indexList,
    int indexLength,
    SeqBitmap*& orBitmap)
{

    orBitmap = new SeqBitmap(*f1[indexList[0]]);
    for (int i = 1; i < indexLength; i++)
        orBitmap->Or(*orBitmap, *f1[indexList[i]]);
}

///////////////////////////////////////////////////////
/// Perform compression on a sequence bitmap. Compression converts as many
/// Bitmap64 bits as possible to Bitmap32 bits, Bitmap32 to Bitmap16, and so
/// on, with the goal of speeding up support counting. We can only get rid
/// of a given bit in a bitmap if it will never be used further on down the
/// tree. The refBitmap parameter should contain a 0 for every bit that
/// is never again used, and a 1 for a bit that is used again. Note that
/// the frequent-1 itemset bitmaps are also compressed so that their bits
/// are still aligned with the sequence bitmap after compression.
///
/// @param refBitmap            Bitmap specifying which bits are compressible
/// @param tempAndBitmap        The sequence bitmap to compress
/// @param returnBitmap         The compressed sequence bitmap
/// @param f1                   Frequent-1 itemset bitmaps prior to compression
/// @param newF1                The compressed versions of the frequent-1
///                                 itemset bitmaps
/// @param indexList            The list of item names for the 1 itemsets that
///                                 are still frequent
/// @param indexLength          The length of indexList
///////////////////////////////////////////////////////
void Compress(
    SeqBitmap* refBitmap,
    SeqBitmap* tempAndBitmap,
    SeqBitmap*& returnBitmap,
    SeqBitmap** f1,
    SeqBitmap** newF1,
    int* indexList,
    int indexLength)
{

    numCompress++;

    // Allocation sizes
    int size4 = 0;
    int size8 = 0;
    int size16 = 0;
    int size32 = 0;
    int size64 = 0;

    // Bitmap position pointers
    int pos4;
    int pos8;
    int pos16;
    int pos32;
    int pos64;

    int i;

    //testFile << " Sepcial Bitmap " << endl;
    //specialBitmap->PrintBitmap(testFile);

    /*    SeqBitmap *refBitmap = new SeqBitmap(*f1[indexList[0]]);
     for (i=1; i < indexLength; i++)
      refBitmap->Or(*refBitmap, *f1[indexList[i]]);

    */
    //testFile << " Reference Bitmap " << endl;
    //refBitmap->PrintBitmap(testFile);

    /* SeqBitmap* specialBitmap = new SeqBitmap(*tempAndBitmap);
     specialBitmap->CreateSpecialBitmap(*tempAndBitmap);
     refBitmap->And(*refBitmap, *specialBitmap);
    */

    /* SeqBitmap* refBitmap = new SeqBitmap(*tempAndBitmap);
     refBitmap->CreateSpecialBitmap(*tempAndBitmap);
     refBitmap->And(*refBitmap, *orBitmap);
    */
    //testFile << " Reference Bitmap (after and)" << endl;
    //refBitmap->PrintBitmap(testFile);



    // Find the allocation sizes
    refBitmap->CountSmaller(size4, size8, size16, size32, size64, summaryFile);

    // Allocate memory for the compressed bitmaps
    for (i = 0; i < indexLength; i++)
    {
        newF1[indexList[i]] =
            new SeqBitmap(size4, size8, size16, size32, size64);
    }

    returnBitmap = new SeqBitmap(size4, size8, size16, size32, size64);

    // summaryFile <<
    // "SPAM::COMPRESS: Outputting old sizes, new sizes, and the current level"
    // << endl;

    // ofstream debugFile;
    // debugFile.open("testOutput.txt");
    // debugFile << "tempANdBitmap"<<endl;
    // tempAndBitmap->PrintBitmap(debugFile);
    // debugFile << "refBitmap"<<endl;
    // refBitmap->PrintBitmap(debugFile);



    // refBitmap->printSizes(summaryFile);
    // summaryFile <<
    // size4 << " " <<
    // size8 << " " <<
    // size16 << " " << size32 << " " << size64 << endl;
    // returnBitmap->printSizes(summaryFile);

    // summaryFile << globalLevel << endl;

    pos4 = 0;
    pos8 = 0;
    pos16 = 0;
    pos32 = 0;
    pos64 = 0;

    // Compress bitmap4
    // pos4 is the first position in bitmap4 free after compress4 is run
    if (refBitmap->_bitmap4 != 0)
        returnBitmap->Compress4(
                  refBitmap, tempAndBitmap,
                  pos4);

    // Compress bitmap8
    // pos8 is the first position in bitmap8 free after compress8 is run
    if (refBitmap->_bitmap8 != 0)
        returnBitmap->Compress8(
                  refBitmap, tempAndBitmap,
                  pos4, pos8);

    // Compress bitmap16
    // pos16 is the first position in bitmap16 free after compress16 is run
    if (refBitmap->_bitmap16 != 0)
        returnBitmap->Compress16(
                  refBitmap, tempAndBitmap,
                  pos4, pos8, pos16);

    // Compress bitmap32
    // pos32 is the first position in bitmap32 free after compress32 is run
    if (refBitmap->_bitmap32 != 0)
        returnBitmap->Compress32(
                  refBitmap,
                  tempAndBitmap,
                  pos4, pos8, pos16, pos32);

    // Compress bitmap64
    // pos64 is the first position in bitmap64 free after compress64 is run
    if (refBitmap->_bitmap64 != 0)
        returnBitmap->Compress64(
                  refBitmap, tempAndBitmap,
                  pos4, pos8, pos16, pos32, pos64);


    // debugFile << "compress bitmap"<<endl;
    // returnBitmap->PrintBitmap(debugFile);



    assert(pos64 == size64*8);
    assert(pos32 == size32*4);
    assert(pos16 == size16*2);
    assert(pos8 == size8);
    assert(pos4*2 == size4);

    // Now start the compression
    for (i = 0; i < indexLength; i++)
    {

        pos4 = 0;
        pos8 = 0;
        pos16 = 0;
        pos32 = 0;
        pos64 = 0;

        // Compress bitmap4
        // pos4 is the first position in bitmap4 free after compress4 is run
        if (refBitmap->_bitmap4 != 0)
            newF1[indexList[i]]->Compress4(refBitmap, f1[indexList[i]],
                                           pos4);

        // Compress bitmap8
        // pos8 is the first position in bitmap8 free after compress8 is run
        if (refBitmap->_bitmap8 != 0)
            newF1[indexList[i]]->Compress8(refBitmap, f1[indexList[i]],
                                           pos4, pos8);

        // Compress bitmap16
        // pos16 is the first position in bitmap16 free after compress16 is run
        if (refBitmap->_bitmap16 != 0)
            newF1[indexList[i]]->Compress16(refBitmap, f1[indexList[i]],
                                            pos4, pos8, pos16);

        // Compress bitmap32
        // pos32 is the first position in bitmap32 free after compress32 is run
        if (refBitmap->_bitmap32 != 0)
            newF1[indexList[i]]->Compress32(refBitmap, f1[indexList[i]],
                                            pos4, pos8, pos16, pos32);

        // Compress bitmap64
        // pos64 is the first position in bitmap64 free after compress64 is run
        if (refBitmap->_bitmap64 != 0)
            newF1[indexList[i]]->Compress64(refBitmap, f1[indexList[i]],
                                            pos4, pos8, pos16, pos32, pos64);


        assert(pos64 == size64*8);
        assert(pos32 == size32*4);
        assert(pos16 == size16*2);
        assert(pos8 == size8);
        assert(pos4*2 == size4);

    }

}



/////////////////////////////////////////////////////////////////////
/// A recursive call that goes down the search lattice to find sequential
///    patterns.
///
/// @param curNode           information about the current node
/////////////////////////////////////////////////////////////////////
void FindSequentialPatterns(TreeNode* curNode)
{
    //cout << curNode->level << " " << testCount << endl;
    testCount++;
    //if (testCount >= 100) {
    // exit(0);
    //}

#ifdef __STATS__

    statistics->startLevelTimer();
    statistics->updateNumNodes(curNode->level, 1);
#endif

    // temp variables to store s-list/i-list of next level
    int* nextLevelSList;
    int* nextLevelIList;
    int nextLevelSLength;
    int nextLevelILength;

    // get node and bitmaps from buffers
    TreeNode* nextNode = nodeBuff[curNode->level + 1];
    SeqBitmap* tempAndBitmap = new SeqBitmap(*curNode->iBitmap);
    SeqBitmap* sBitmap = new SeqBitmap(*curNode->iBitmap);
    //SeqBitmap* returnBitmap = 0;

    // SeqBitmap* tempAndBitmap = tempBitmapBuff[curNode->level];
    // SeqBitmap* sBitmap = sBitmapBuff[curNode->level];

    int count = 0;     // count the support of a bitmap
    int i;       // counter

#ifdef __COMPRESSION__

    bool prevComp = false;  // a flag to tell if we compress the previous node
    int j;
    int memSize;
#endif

    // -- s-step
    // create the s-bitmap, using the post-processing step of setting the
    // first 1 in each bit slice to 0,
    // and all subsequent bits to 1
    sBitmap->CreateSBitmap(*curNode->iBitmap);

    // find s-extensions for the next node
    nextLevelSList = curNode->sList + curNode->sLength;
    nextLevelIList = curNode->iList + curNode->iLength;
    nextLevelSLength = 0;

    // for each possible s-extension from this level
    for (i = 0; i < curNode->sLength; i++)
    {
        // AND the post-processed s-step bitmap with the candidate
        // frequent-1 itemset
        tempAndBitmap->And(*sBitmap, *curNode->f1[curNode->sList[i]]);
        count = tempAndBitmap->Count();

        // If the AND'd result is frequent, specify in curNode->countList that
        // the s-extended sequence should be traversed further
        if (count >= minSup)
        {
            curNode->countList[nextLevelSLength].count = count;
            curNode->countList[nextLevelSLength].index = curNode->sList[i];
            nextLevelSLength++;
        }

    }

#ifdef __STATS__
    statistics->updateSExt(curNode->level, nextLevelSLength);
#endif


#ifdef ___REORDERING___
    // Sort nextLevelSList based on curNode->countList so that extensions
    // with higher support are traversed first
    qsort((void *) curNode->countList, (size_t)nextLevelSLength,
          sizeof(int) + sizeof(int), compare );
#endif

    // Set values in nextLevelSList based on the countList indices
    for (i = 0;i < nextLevelSLength;i++)
        nextLevelSList[i] = curNode->countList[i].index;

    // create i-extensions for the next node
    for (i = 0;i < nextLevelSLength - 1;i++)
        nextLevelIList[i] = nextLevelSList[i + 1];

    // initialize nextNode
    nextNode->level = curNode->level + 1;
    nextNode->sList = nextLevelSList;
    nextNode->sLength = nextLevelSLength;
    nextNode->f1 = curNode->f1;
    nextNode->f1Name = curNode->f1Name;
    nextNode->f1Size = curNode->f1Size;
    nextNode->compress = false;
#ifdef __COMPRESSION__
    // Do we want to compress every 5 levels, 10 levels, or 15 levels?
    // Not sure of optimal times to compress.
    //if ((nextNode->sLevel % 5) == 1) { nextNode->compress = true; }
    //if ((nextNode->sLevel % 10) == 1) { nextNode->compress = true; }
    if ((nextNode->sLevel % 15) == 1)
    {
        nextNode->compress = true;
    }
#ifdef __ALLCOMPRESS__
    nextNode->compress = true;
#endif
#else

    nextNode->compress = false;
#endif


    // for output
    if (outputSeq)
        sequenceLength++;

#ifdef __COMPRESSION__
#ifdef __STATS__

    statistics->startCompTimer();
#endif

    SeqBitmap *orBitmap = 0;
    SeqBitmap *specialBitmap = 0;


    if (curNode->compress && nextLevelSLength > 0)
    {
        CreateOrBitmap(
            curNode->f1,
            nextLevelSList,
            nextLevelSLength,
            orBitmap);

        specialBitmap = new SeqBitmap(*tempAndBitmap);
    }

#ifdef __STATS__
    statistics->stopCompTimer(curNode->level);
#endif
#endif

    // perform s-step
    for (i = 0; i < nextLevelSLength; i++)
    {
        // Update the sLevel for the next node
        nextNode->sLevel = curNode->sLevel + 1;

        // i-list only contains extensions after the current node
        nextNode->iList = nextLevelIList + i;
        nextNode->iLength = nextLevelSLength - 1 - i;
        tempAndBitmap->And(*sBitmap, *curNode->f1[nextLevelSList[i]]);

        if (outputSeq)
            count = tempAndBitmap->Count();

#ifdef __COMPRESSION__
#ifdef __STATS__

        statistics->startCompTimer();
#endif
        //int nextLevelSLengthLocal;
        //int * nextLevelSListLocal;
        //SeqBitmap **tempF1Local;
        memSize = tempAndBitmap->memSize();
        // compress the data when condition holds
        if ((memSize >= minCompSize) && curNode->compress)
        {

            prevComp = true;

            // update the current frequent-one itemset buffer
            nextNode->f1 = curNode->f1 + curNode->f1Size;

            specialBitmap->CreateCBitmap(*tempAndBitmap);
            specialBitmap->And(*specialBitmap, *orBitmap);

            // Jay: For debugging of compression
            globalLevel = curNode->level;



#ifdef __STATS__

            statistics->updateNumComp(curNode->level, 1);
#endif
            //nextLevelSLengthLocal = nextLevelSLength;
            //nextLevelSListLocal = new int[nextLevelSLength];
            //for (j = 0; j < nextLevelSLength; j++) {
            // nextLevelSListLocal[j] = nextLevelSList[j];
            //}

            //tempF1Local = new SeqBitmap*[nextLevelSLength];
            //for (j = 0; j < nextLevelSLength; j++) {
            // tempF1Local[j] = nextNode->f1[j];
            //}

            Compress(
                specialBitmap,
                tempAndBitmap,
                returnBitmap,
                curNode->f1,
                nextNode->f1,
                nextLevelSList,
                nextLevelSLength);

            nextNode->iBitmap = returnBitmap;
        }
        else
        {
            // If compression was done for the last extension,
            // reset the values for nextNode
            if (prevComp)
            {
                prevComp = false;
                nextNode->f1 = curNode->f1;
            }
            nextNode->iBitmap = tempAndBitmap;
        }
#ifdef __STATS__
        statistics->stopCompTimer(curNode->level);
#endif
#else

        nextNode->iBitmap = tempAndBitmap;
#endif

        // output sequence
        if (outputSeq)
        {
            elementSize[sequenceLength] = 0;
            sequentialPatterns[sequenceLength][0] =
                curNode->f1Name[nextLevelSList[i]];
            LogSequence(count);
        }

#ifdef __STATS__
        statistics->stopLevelTimer(curNode->level);
#endif

        // Recurse on the next node
        FindSequentialPatterns(nextNode);

#ifdef __STATS__

        statistics->startLevelTimer();
#endif

#ifdef __COMPRESSION__
#ifdef __STATS__

        statistics->startCompTimer();
#endif

        if (prevComp)
        {
            // Deallocate/delete returnBitmap
            returnBitmap->Deallocate();
            delete returnBitmap;

            // Deallocate the F1s
            //assert(nextLevelSLengthLocal == nextLevelSLength);

            for (j = nextNode->f1Size - 1; j >= 0; j--)
            {
                if (nextNode->f1[j] != 0)
                {
                    assert(nextNode->f1[j] != (SeqBitmap *)0xfdfdfdfd);
                    nextNode->f1[j]->Deallocate();
                    delete nextNode->f1[j];
                }
                nextNode->f1[j] = 0;
            }

            for (j = 0; j < nextNode->f1Size; j++)
                assert(nextNode->f1[j] == 0);
        }
#ifdef __STATS__
        statistics->stopCompTimer(curNode->level);
#endif
#endif

    }

#ifdef __COMPRESSION__
#ifdef __STATS__
    statistics->startCompTimer();
#endif

    if (curNode->compress && nextLevelSLength != 0)
    {
        // Deallocate specialBitmap and orBitmap in reverse order of allocation
        specialBitmap->Deallocate();
        orBitmap->Deallocate();
        delete orBitmap;
        delete specialBitmap;
    }
#ifdef __STATS__
    statistics->stopCompTimer(curNode->level);
#endif
#endif

    // for output
    if (outputSeq)
        sequenceLength--;


    // -- i-step
    // find i-extensions
    nextLevelIList = curNode->iList + curNode->iLength;
    nextLevelILength = 0;

    // For each possible i-step
    for (i = 0; i < curNode->iLength; i++)
    {
        // Update the sLevel for the next node
        // Not increasing sLevel, just copy it over
        nextNode->sLevel = curNode->sLevel;

        tempAndBitmap->And(*curNode->iBitmap, *curNode->f1[curNode->iList[i]]);
        count = tempAndBitmap->Count();

        // If the i-step was frequent, the i-step sequence is considered
        // a candidate
        if (count >= minSup)
        {
            curNode->countList[nextLevelILength].count = count;
            curNode->countList[nextLevelILength].index = curNode->iList[i];
            nextLevelILength++;
        }
    }

#ifdef __STATS__
    statistics->updateIExt(curNode->level, nextLevelILength);
#endif
#ifdef ___REORDERING___
    // Sort nextLevelSList based on curNode->countList
    qsort( (void *) curNode->countList, (size_t)nextLevelILength,
           sizeof(int) + sizeof(int), compare );
#endif

    // Set values in nextLevelSList based on the reordered indices
    for (i = 0;i < nextLevelILength;i++)
        nextLevelIList[i] = curNode->countList[i].index;


    // i-step
    // for output
    if (outputSeq)
        elementSize[sequenceLength]++;


    for (i = 0; i < nextLevelILength; i++)
    {
        nextNode->iList = nextLevelIList + i + 1;
        nextNode->iLength = nextLevelILength - i - 1;

        tempAndBitmap->And(*curNode->iBitmap, *curNode->f1[nextLevelIList[i]]);
        if (outputSeq)
            count = tempAndBitmap->Count();


        // not doing compression for i-step
#ifdef __COMPRESSION__

        memSize = tempAndBitmap->memSize();

        if ((memSize >= minCompSize) && curNode->compress && false)
        {
            prevComp = true;

            // update the current frequent-one itemset buffer
            nextNode->f1 = curNode->f1 + curNode->f1Size;

            int j;
            // find the indices of f1's which have to be compressed
            for (j = 0; j < nextNode->f1Size; j++)
                indexExists[j] = false;

            // copy indicies from s-list
            for (j = 0; j < nextLevelSLength; j++)
            {
                tempIndexList[j] = nextLevelSList[j];
                indexExists[nextLevelSList[j]] = true;
            }
            int indexLength = nextLevelSLength;

            // copy indices from i-list but making sure that it doesn't already
            // exist in the s-list
            for (j = 0; j < nextNode->iLength; j++)
            {
                if (!indexExists[nextNode->iList[j]])
                {
                    tempIndexList[indexLength] = nextNode->iList[j];
                    indexLength++;
                }
            }

            // Compress(tempAndBitmap, returnBitmap, curNode->f1, nextNode->f1,
            // tempIndexList, indexLength);
            nextNode->iBitmap = tempAndBitmap;
        }
        else
        {

            if (prevComp)
            {
                prevComp = false;
                nextNode->f1 = curNode->f1;
            }
            nextNode->iBitmap = tempAndBitmap;

        }
#else
        nextNode->iBitmap = tempAndBitmap;
#endif


        // output
        if (outputSeq)
        {
            sequentialPatterns[sequenceLength][elementSize[sequenceLength]] =
                curNode->f1Name[nextLevelIList[i]];
            LogSequence(count);
        }

#ifdef __STATS__
        statistics->stopLevelTimer(curNode->level);
#endif

        // Recurse on the next node
        FindSequentialPatterns(nextNode);
#ifdef __STATS__

        statistics->startLevelTimer();
#endif

#ifdef __COMPRESSION__

        if (prevComp)
        {
            delete returnBitmap;
            for (j = 0; j < nextNode->f1Size; j++)
            {
                if (nextNode->f1[j] != 0)
                    delete nextNode->f1[j];
                nextNode->f1[j] = 0;
            }
        }
#endif

    }

    // for output
    if (outputSeq)
        elementSize[sequenceLength]--;


    // Deallocate the SeqBitmaps used in this node in reverse order of
    // allocation
    sBitmap->Deallocate();
    tempAndBitmap->Deallocate();
    delete tempAndBitmap;
    delete sBitmap;

#ifdef __STATS__

    statistics->stopLevelTimer(curNode->level);
#endif

}


/////////////////////////////////////////////////////////////////////
/// Start the mining algorithm by generating the initial TreeNode to
/// start recursing from.
///
/// @param info              information about the data set
/////////////////////////////////////////////////////////////////////
void StartMining(DatasetInfo* info)
{
#ifdef __STATS__
    statistics->startLevelTimer();
    statistics->updateNumNodes(0, 1);
    statistics->updateSExt(0, info->f1Size);
#endif

    int i;
    int count;

    TreeNode* curNode = nodeBuff[0];
    curNode->f1 = info->f1Buff;
    curNode->f1Name = info->f1NameBuff;
    curNode->f1Size = info->f1Size;
    curNode->sList = info->sListBuff;
    curNode->sLength = info->f1Size;
    curNode->countList = info->countBuff;
    curNode->level = 0;
    curNode->sLevel = 0;
    curNode->compress = false;


    for (i = 0; i < info->f1Size; i++)
    {
        count = info->f1Buff[i]->Count();
        curNode->countList[i].count = count;
        curNode->countList[i].index = i;

    }

#ifdef ___REORDERING___
    // Sort SList based on curNode->countList
    qsort( (void *) curNode->countList, (size_t)info->f1Size,
           sizeof(int) + sizeof(int), compare );
#endif

    for (i = 0; i < info->f1Size; i++)
    {
        info->sListBuff[i] = curNode->countList[i].index;
        info->iListBuff[i] = curNode->countList[i].index;
    }


    // call FindSequentialPatterns
    for (i = 0; i < info->f1Size; i++)
    {

        // output sequence
        if (outputSeq)
        {
            count = info->f1Buff[info->sListBuff[i]]->Count();
            elementSize[sequenceLength] = 0;
            sequentialPatterns[sequenceLength][0] =
                info->f1NameBuff[info->sListBuff[i]];
            LogSequence(count);
        }

        curNode->iList = info->iListBuff + i + 1;
        curNode->iLength = info->f1Size - i - 1;
        curNode->iBitmap = info->f1Buff[info->sListBuff[i]];

#ifdef __STATS__

        statistics->stopLevelTimer(0);
#endif

        FindSequentialPatterns(curNode);

#ifdef __STATS__

        statistics->startLevelTimer();
#endif

    }

#ifdef __STATS__
    statistics->stopLevelTimer(0);
#endif

}


void PrintError()
{
    cerr << "\nUsage: spam -sup <minSup> [-fn <infile>] [-stdin] [-ascii] [-str]\n";
    cerr << "             [-outFile <outfile>] [-stdout]\n\n";

    // (See below comment re emptRatio, compLevel, and minSize)
    // cerr << "             [-emptRatio e] [-compLevel c] [-minSize m]\n\n";
    cerr << "    minSup    - The minimum support (between 0.0 and 1.0)\n";
    cerr << "    infile    - The data file to read in (see below for specifications)\n";
    cerr << "    stdin     - Use this flag if the data should be read in from stdin.\n";
    cerr << "                Must use if -fn is not specified.\n";
    cerr << "                Overrides any file specified via -fn\n";
    cerr << "    ascii     - Use this flag if your input is ASCII text;\n";
    cerr << "                otherwise SPAM assumes it is in a binary file format\n";
    cerr << "    str       - Use this flag if your input is a list of strings\n";
    cerr << "                representing customers, transactions, and items\n";
    cerr << "                (see documentation for full file-format description)\n";
    cerr << "    outfile   - The file to place the output in. If -outFile and -stdout\n";
    cerr << "                are not present, no output will be produced\n";
    cerr << "    stdout    - Use this flag if you want the output to go to stdout.\n";
    cerr << "                Overrides any file specified via -outFile\n\n";
    cerr << "See documentation for examples of how to use SPAM with various data formats.\n";

    // emptRatio, minSize, and compLevel are not needed to just run the
    // algorithm; for simplicity's sake they are not documented in the
    // usage instructions
    /*
    cerr << "    emptRatio - The percentage (0.0 to 1.0) of empty spaces";
    cerr << " a bitmap \n";
    cerr << "                  must have to warrant compression\n";
    cerr << "    minSize   - The size (in # of shorts) a bitmap must be";
    cerr << " reduced to in \n";
    cerr << "                  order to stop compression\n";
    cerr << "    compLevel - do compression every c levels \n";
    */
    exit(0);
}


int main(int argc, char **argv){
    // start timing
    clock_t programStart = clock();

    // -- command prompt variables
    bool isBinaryFile;    // whether the input file is a binary file
    double minSupPercent = 0;  // user-defined min sup as percentage
    char* inFilename = 0;
    char outFilename[50];
    bool fnSpecified = false;
    bool supSpecified = false;
    bool stdinSpecified = false;
    outputSeq = false;      //flag for the output sequence

    // -- variables for timing
    clock_t programEnd;
    clock_t miningStart, miningEnd; 
    double programDuration;
    double miningDuration;

    DatasetInfo* info;

    // -- summaryFile
    //ofstream summaryFile;

    // --- Parse the command line
    // default value
    strcpy(outFilename, "output.txt");
    emptySpaceRatio = 0.8;
    minCompSize = 30;
    compLevel = 1;
    isBinaryFile = true;
    isStringFile = false;

    // parse input
    int i = 1;
    while (i < argc)
    {

        if (strcmp(argv[i], "-fn") == 0)
        {

            // input filename
            i++;
            if (i == argc)
                PrintError();
            else
            {
                inFilename = argv[i];
                fnSpecified = true;
            }

        }
        else if (strcmp(argv[i], "-sup") == 0 )
        {

            // support percentage
            i++;
            if (i == argc)
                PrintError();
            else
            {
                minSupPercent = atof(argv[i]);
                supSpecified = true;
            }

        }
        else if (strcmp(argv[i], "-emptRatio") == 0 )
        {

            // empty space ratio
            i++;
            if (i == argc)
                PrintError();
            else
                emptySpaceRatio = atof(argv[i]);

        }
        else if (strcmp(argv[i], "-minSize") == 0)
        {

            // minimum compression size
            i++;
            if (i == argc)
                PrintError();
            else
                minCompSize = atoi(argv[i]);

        }
        else if (strcmp(argv[i], "-compLevel") == 0)
        {

            // comp level
            i++;
            if (i == argc)
                PrintError();
            else
                compLevel = atoi(argv[i]);

        }
        else if (strcmp(argv[i], "-outFile") == 0)
        {

            // output file name
            i++;
            if (i == argc)
                PrintError();
            else
            {
                strcpy(outFilename, argv[i]);
                outputSeq = true;
            }

        }
        else if (strcmp(argv[i], "-ascii") == 0)
        {

            // whether the input file is ascii
            isBinaryFile = false;

        }
        else if (strcmp(argv[i], "-str") == 0)
        {
            isBinaryFile = false;
            isStringFile = true;
        }
        else if (strcmp(argv[i], "-stdin") == 0)
        {
            stdinSpecified = true;
        }
        else if (strcmp(argv[i], "-stdout") == 0)
        {
            stdoutSpecified = true;
            outputSeq = true;
        }
        else
        {
            PrintError();
        }

        i++;
    }

    if (!supSpecified)
    {
        cout << "Must specify minimum support" << endl;
        PrintError();
    }
    // If the filename was not specified or stdin was specified,
    // attempt to read in from stdin
    if (!fnSpecified || stdinSpecified)
        inFilename = 0;

    if (!fnSpecified && !stdinSpecified)
    {
        // The user didn't select an input file
        // Let them know that we are reading from stdin by default
        cout << "No input data source selected; use either -stdin or -fn" << endl;
        PrintError();
    }

    if (stdinSpecified && isBinaryFile)
    {
        cout << "Cannot use binary files with -stdin; use -ascii or -str" << endl;
        PrintError();
    }

    bool suppressOutput = false;
    // If either reading from stdin or writing to stdout, we
    // want to suppress all extraneous output to the screen
    if (stdinSpecified || stdoutSpecified)
        suppressOutput = true;

    if (isBinaryFile == true)
    {
        if (!suppressOutput)
        {
            cout << "Input file will be interpreted as binary, ";
            cout << "if SPAM crashes use the -ascii flag\n";
        }
    }
    else
    {
        if (!suppressOutput)
            cout << "Input file will be interpreted as ASCII...\n";
    }

    // --- Read in data
    if (!suppressOutput)
        cout << "Reading Data... " << endl;

    info = ReadDataset(isBinaryFile,
                       isStringFile,
                       inFilename,
                       minSupPercent,
                       custStrMap,
                       transStrMap,
                       itemStrMap);
    if (info == 0)
    {
        cout << "Error reading dataset" << endl;
        exit(1);
    }

    // --- Open output file
    if (outputSeq && !stdoutSpecified)
        outFile.open(outFilename);
    summaryFile.open("summary.txt");

    if (info->f1Size > 0)
    {

        // --- Preparing for mining sequential patterns
        minSup = info->minSup;
        totalCust = info->custCount;

        // build sbitmap process table
        SeqBitmap::Init();


        // initialize buffers

        // max depth of the tree is the maximum number of transaction
        // a customer (s-tree) has plus
        // the size of f1 (i-tree)
        int bufSize = info->f1Size * info->maxCustTrans + 1;
        nodeBuff = new TreeNode * [bufSize];
        //tempBitmapBuff = new SeqBitmap*[bufSize];
        //sBitmapBuff = new SeqBitmap*[bufSize];



        for (i = 0; i < bufSize; i++)
        {
            nodeBuff[i] = new TreeNode();
            //tempBitmapBuff[i] =
            // new SeqBitmap(info->bitmapSizes[0], info->bitmapSizes[1],
            // info->bitmapSizes[2], info->bitmapSizes[3],
            // info->bitmapSizes[4]);
            //sBitmapBuff[i] = new SeqBitmap(info->bitmapSizes[0],
            // info->bitmapSizes[1], info->bitmapSizes[2],
            // info->bitmapSizes[3], info->bitmapSizes[4]);
        }

        // In doing compression, we have to know what f1 to compress, and
        // hence we have to combine the i-list and s-list. These two buffers
        // are for that purpose.
        tempIndexList = new int[info->f1Size * 2];
        indexExists = new bool[info->f1Size];

        // If output sequence,
        // initialize array for printing sequential patterns
        if (outputSeq)
        {
            sequentialPatterns = new int * [info->maxCustTrans];
            elementSize = new int[info->maxCustTrans];
            for (i = 0; i < info->maxCustTrans; i++)
            {
                sequentialPatterns[i] = new int[info->f1Size];
                elementSize[i] = 0;
            }
            sequenceLength = 0;
        }


#ifdef __STATS__
        statistics = new Stats(bufSize);
#endif

        // -- Mine sequential patterns
        if (!suppressOutput)
            cout << "Running the algorithm..." << endl;
        miningStart = clock();
        StartMining(info);
        miningEnd = clock();
        if (!suppressOutput)
            cout << "Done." << endl;

#ifdef __STATS__

        statistics->printStats(summaryFile);
#endif

#ifdef __STATS__

        delete statistics;
#endif

        // deallocate memory
        for (i = 0; i < bufSize; i++)
        {
            delete nodeBuff[i];
            //delete tempBitmapBuff[i];
            //delete sBitmapBuff[i];
        }

        delete [] nodeBuff;
        //delete [] tempBitmapBuff;
        //delete [] sBitmapBuff;
        delete [] tempIndexList;
        delete [] indexExists;

        delete custStrMap;
        delete transStrMap;
        delete itemStrMap;


        if (outputSeq)
        {

            for (i = 0; i < info->maxCustTrans; i++)
                delete[] sequentialPatterns[i];
            delete[] sequentialPatterns;
            delete[] elementSize;
        }

        SeqBitmap::Destroy();

    }
    else
    {

        miningStart = 0;
        miningEnd = 0;

    }

    // -- time calculation
    miningDuration = (miningEnd - miningStart) / (double)CLOCKS_PER_SEC;

    programEnd = clock();
    programDuration = (programEnd - programStart) / (double)CLOCKS_PER_SEC;

    summaryFile << "Number of customer: " << info->custCount << endl;
    summaryFile << "Minimum support: " << minSup << " ( " << minSupPercent
    << " )" << endl;
    summaryFile << "Mining Duration:" << miningDuration << endl;
    summaryFile << "Program Duration: " << programDuration << endl << endl;

    summaryFile << "Number of Compression: " << numCompress << endl;
    summaryFile << "Number of Ors: " << SeqBitmap::_countOr << endl;
    summaryFile << "Number of Ands: " << SeqBitmap::_countAnd << endl;
    summaryFile << "Number of Count: " << SeqBitmap::_countCount << endl;
    summaryFile << "Number of CountZeros: " <<
    SeqBitmap::_countCountZeros << endl;
    summaryFile << "Number of CountSmaller: " <<
    SeqBitmap::_countCountSmaller << endl;
    summaryFile << "Number of CreateSBitmap: " <<
    SeqBitmap::_countCreateSBitmap << endl;
    summaryFile << "Number of CreateCBitmaps: " <<
    SeqBitmap::_countCreateCBitmap << endl;

    // deallocate memory
    if (outputSeq)
        outFile.close();
    summaryFile.close();

    // Deallocate initial f1 buff in reverse order of allocation
    for (i = info->f1Size - 1; i >= 0; i--)
        info->f1Buff[i]->Deallocate();

    delete info;

    // Do final deallocation of SeqBitmap memory
    SeqBitmap::MemDealloc();
    return 0;
}

/** @} */


/////////////////////////////////////////////////////////////////////
/// @mainpage SPAM Code Documentation
/// \image html SpamLogo.jpg
///
/// \section ContactSection Contact
/// - Jay Ayres (kja9@cornell.edu)
/// - Manuel Calimlim (calimlim@cs.cornell.edu)
/// - Johannes Gehrke (johannes@cs.cornell.edu)
///
/// \section DownloadSection Download
/// - Main webpage at
///   http://himalaya-tools.sourceforge.net/Spam
/// - Older webpage at
///   http://www.cs.cornell.edu/database/himalaya/spam/spam.htm
///
/// \section LinuxCompilationSection Linux Compilation
/// -# ./configure
/// -# make
///     - executable is created as 'src/spam'
/// -# make install
///     - will copy the executable to /usr/local/bin
///     - make sure you have the right permission to install there
///
/// \section WindowsCompilationSection Windows Compilation
/// -# Open Spam.sln in Visual Studio .NET (previous versions of Visual Studio not supported)
/// -# Build the program
///
/// \section DirectoryStructureSection Directory Structure
/// <table border=1 cellpadding=5 cellspacing=0>
/// <tr>
/// <td>admin/</td>
/// <td>Contains config files for compiling.  Should
///     not be altered.</td>
/// </tr>
/// <tr>
/// <td>datasets/</td>
/// <td>Contains several datasets for testing</td>
/// </tr>
/// <tr>
/// <td>src/</td>
/// <td>Contains all of source code for SPAM</td>
/// </tr>
/// <tr>
/// <td>src/Bitmap4.h
/// <br>src/Bitmap4.cpp
/// <br>src/Bitmap8.h
/// <br>src/Bitmap8.cpp
/// <br>src/Bitmap16.h
/// <br>src/Bitmap16.cpp
/// <br>src/Bitmap32.h
/// <br>src/Bitmap32.cpp
/// <br>src/Bitmap64.h
/// <br>src/Bitmap64.cpp
/// </td>
/// <td>
/// Vertical representations of the data. They handle both uncompressed
///  and compressed data.
///  <p>
/// Each bitmap can represent customers with up to x transactions since it
///  allocates exactly x bits per customer
/// </td>
/// </tr>
/// <tr>
/// <td>src/DatasetInfo.h</td>
/// <td>A class for representing the info gathered from the dataset</td>
/// </tr>
/// <tr>
/// <td>src/FileInput.cpp</td>
/// <td>The functions necessary for reading in datasets.</td>
/// </tr>
/// <tr>
/// <td>src/ResizableArray.h</td>
/// <td>A class containing a resizable array data structure</td>
/// </tr>
/// <tr>
/// <td>src/SeqBitmap.h
/// <br>src/SeqBitmap.cpp</td>
/// <td>A representation of a sequence (or an item)</td>
/// </tr>
/// <tr>
/// <td>src/Spam.cpp</td>
/// <td>Main file with most of the algorithmic code</td>
/// </tr>
/// <tr>
/// <td>src/Stats.h
/// <br>src/Stats.cpp</td>
/// <td>Collects statistics about SPAM as it is run</td>
/// </tr>
/// <tr>
/// <td>src/StringMap.h</td>
/// <td>A class containing a data structure that allows for two way mapping between ints and strings</td>
/// </tr>
/// <tr>
/// <td>src/Tables.h</td>
/// <td>For gathering the lookup tables in one place</td>
/// </tr>
/// <tr>
/// <td>src/TreeNode.h</td>
/// <td>Class for representing nodes in the search tree</td>
/// </tr>
/// <tr>
/// <td>INSTALL</td>
/// <td>Generic installation instructions</td>
/// </tr>
/// <tr>
/// <td>spam.{kdevprj,kdevses}</td>
/// <td>KDevelop project files for Linux</td>
/// </tr>
/// <tr>
/// <td>Spam.{sln,vcproj}</td>
/// <td>Visual Studio .NET project files for Windows</td>
/// </tr>
/// <tr>
/// <td>README</td>
/// <td>Pointer to this page</td>
/// </tr>
/// <tr>
/// <td>test</td>
/// <td>Contains perlscripts and executables to test SPAM</td>
/// </tr>
/// </table>
///
/// \section UsageSection Program Usage
/// <pre>
/// Usage: spam -sup &lt;minSup&gt; [-fn &lt;infile&gt;] [-stdin] [-ascii] [-str]
///             [-outFile &lt;outfile&gt;] [-stdout]
///
///    minSup    - The minimum support (between 0.0 and 1.0)
///    infile    - The data file to read in (see below for specifications)
///    stdin     - Use this flag if the data should be read in from stdin.
///                Must use if -fn is not specified.
///                Overrides any file specified via -fn
///    ascii     - Use this flag if your input is ASCII text;
///                otherwise SPAM assumes it is in a binary file format
///    str       - Use this flag if your input is a list of strings
///                representing customers, transactions, and items
///                (see documentation for full file-format description)
///    outfile   - The file to place the output in. If -outFile and -stdout
///                are not present, no output will be produced
///    stdout    - Use this flag if you want the output to go to stdout.
///                Overrides any file specified via -outFile
///
/// There are three input data formats:
/// 1. ASCII numbers (use -ascii)
///    This data format is ASCII text with each line containing the customer ID,
///    the transaction ID, and the item ID separated by spaces. The data must be
///    sorted in ascending order first by cust ID, then by trans ID, then by item ID.
///    Note that SPAM has the limitation that transactions can contain no more than
///    64 items.
///
/// 2. ASCII strings (use -str)
///    This data format is also ASCII text, but each customer, transaction, and item
///    is an actual string instead of a number. Since the strings may have spaces,
///    the customer, transaction, and item must be separated by newline characters
///    instead of by spaces as for format #1. This option should not be used with
///    extremely large files because the input and output is slow compared to #1.
///
/// 3. Binary file (don't use -ascii or -str)
///    This data format is present to support AssocGen-generated data files. See
///    the perlscripts in the test directory for information on how to generate
///    binary files.
///
///
/// New in release 1.3: -stdin and -stdout:
/// Now you can have files come in through standard input and output go through
/// standard output. Note that SPAM does not support having binary files come
/// in through stdin.
/// </pre>
///
/// \section TestingSection Program Testing
/// All datasets used by SPAM were generated by the IBM AssocGen synthetic
/// data generator. Several sample datasets are included in the datasets
/// directory, and the AssocGen executable can be used along with the
/// perlscripts in the test directory to generate custom datasets with
/// varying parameters. Please view the perlscripts for instructions on
/// how to use them.
///
/////////////////////////////////////////////////////////////////////


