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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "StringMap.h"
#include "SeqBitmap.h"
#include "TreeNode.h"
#include "DatasetInfo.h"
#include "Stats.h"
#include "Bitmaps_cuda.h"

using namespace std;

/// @defgroup GlobalVariables Global Variables
/// Global variables
/** @{ */

// Support count of the current node
CountInfo* TreeNode::countList = 0;

ofstream summaryFile;

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
}

void LogFileSequence(const int c)
{
    // Output the sequence to the outFile

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
}

/////////////////////////////////////////////////////////////////////
/// A recursive call that goes down the search lattice to find sequential
///    patterns.
///
/// @param curNode           information about the current node
/////////////////////////////////////////////////////////////////////
void FindSequentialPatterns(TreeNode* curNode)
{

    // temp variables to store s-list/i-list of next level
    int* nextLevelSList;
    int* nextLevelIList;
    int nextLevelSLength;
    int nextLevelILength;

    // get node and bitmaps from buffers
    TreeNode* nextNode = nodeBuff[curNode->level + 1];
	SeqBitmap* tempAndBitmap = new SeqBitmap(*curNode->iBitmap);
    SeqBitmap* sBitmap = new SeqBitmap(*curNode->iBitmap);

    int count = 0;  // count the support of a bitmap
    int i;			// counter

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

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

    // for output
    if (outputSeq)
        sequenceLength++;

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

        nextNode->iBitmap = tempAndBitmap;

        // output sequence
        if (outputSeq)
        {
            elementSize[sequenceLength] = 0;
            sequentialPatterns[sequenceLength][0] =
                curNode->f1Name[nextLevelSList[i]];
            LogSequence(count);
        }

        // Recurse on the next node
        FindSequentialPatterns(nextNode);
    }

    // for output
    if (outputSeq)
        sequenceLength--;

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

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

        nextNode->iBitmap = tempAndBitmap;

        // output
        if (outputSeq)
        {
            sequentialPatterns[sequenceLength][elementSize[sequenceLength]] =
                curNode->f1Name[nextLevelIList[i]];
            LogSequence(count);
        }

        // Recurse on the next node
        FindSequentialPatterns(nextNode);
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
}

/////////////////////////////////////////////////////////////////////
/// Start the mining algorithm by generating the initial TreeNode to
/// start recursing from.
///
/// @param info              information about the data set
/////////////////////////////////////////////////////////////////////
void StartMining(DatasetInfo* info)
{
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

        FindSequentialPatterns(curNode);
    }
}

void PrintError()
{
    cerr << "\nUsage: spam -sup <minSup> [-fn <infile>] [-stdin] [-ascii] [-str]\n";
    cerr << "             [-outFile <outfile>] [-stdout]\n\n";

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

    exit(0);
}

int main(int argc, char **argv)
{
	cudaDeviceReset();

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
    outputSeq = false;

    // -- variables for timing
    clock_t programEnd;
    clock_t miningStart, miningEnd;
    double programDuration;
    double miningDuration;

    DatasetInfo* info;

    // --- Parse the command line
    // default value
    strcpy(outFilename, "output.txt");
    emptySpaceRatio = 0.8;
    minCompSize = 30;
    compLevel = 1;
    isBinaryFile = true;
    isStringFile = false;

#pragma region _PARSE_
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

#pragma endregion 
    
	if (info->f1Size > 0)
    {

        // --- Preparing for mining sequential patterns
        minSup = info->minSup;
        totalCust = info->custCount;

        // build sbitmap process table
		////////////////////////////////
        SeqBitmap::Init();
		////////////////////////////////
        // initialize buffers

        // max depth of the tree is the maximum number of transaction
        // a customer (s-tree) has plus
        // the size of f1 (i-tree)
        int bufSize = info->f1Size * info->maxCustTrans + 1;
        nodeBuff = new TreeNode * [bufSize];

        for (i = 0; i < bufSize; i++)
            nodeBuff[i] = new TreeNode();


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

        // -- Mine sequential patterns
        if (!suppressOutput)
            cout << "Running the algorithm..." << endl;
        
		miningStart = clock();

		// In this function i will start use the GPU
        StartMining(info);
        
		miningEnd = clock();
        
		if (!suppressOutput)
            cout << "Done." << endl;

        // deallocate memory
        for (i = 0; i < bufSize; i++)
            delete nodeBuff[i];

        delete [] nodeBuff;
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

	summaryFile << "-- GPU --" << endl;
    summaryFile << "Number of customer: " << info->custCount << endl;
    summaryFile << "Minimum support: " << minSup << " ( " << minSupPercent << " )" << endl;
    summaryFile << "Mining Duration:" << miningDuration << endl;
    summaryFile << "Program Duration: " << programDuration << endl << endl;
    summaryFile << "Number of Compression: " << numCompress << endl;
    summaryFile << "Number of Ors: " << SeqBitmap::_countOr << endl;
    summaryFile << "Number of Ands: " << SeqBitmap::_countAnd << endl;
    summaryFile << "Number of Count: " << SeqBitmap::_countCount << endl;
    summaryFile << "Number of CountZeros: 0" << endl;
    summaryFile << "Number of CountSmaller: 0" << endl;
    summaryFile << "Number of CreateSBitmap: " << SeqBitmap::_countCreateSBitmap << endl;
    summaryFile << "Number of CreateCBitmaps: 0" << endl;
    

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
