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
/// FileInput.cpp
/////////////////////////////////////////////////////////////////////
//#define ___PREFIXSPAN___

#include <fstream>
#include <math.h>
#include <stdio.h>
#include "SeqBitmap.h"
#include "Bitmap4.h"
#include "Bitmap8.h"
#include "Bitmap16.h"
#include "Bitmap32.h"
#include "Bitmap64.h"
#include "DatasetInfo.h"
#include "ResizableArray.h"
#include "StringMap.h"
#include <iostream>

#define FILEERR_64TRANSACTIONS 1
#define FILEERR_NOTFOUNDBINARY 2
#define FILEERR_NOTFOUNDASCII 3

/////////////////////////////////////////////////////////////////////
/// @addtogroup FileInputGroup File Input Functions
/// It contains all the functions necessary for reading in datasets.
/** @{ */

//prototypes
void PrintFileReadError(int errorType);
void IncArraySize(int*& array, int oldSize, int newSize);
DatasetInfo* ReadDataset(
                         bool isBinaryFile,
                         bool isStringFile,
                         char *filename,
                         double minSupPercent,
                         StringMap *&custStrMap,
                         StringMap *&transStrMap,
                         StringMap *&itemStrMap);

bool ReadBinary(
                char* filename,
                int* cids,
                int* tids,
                int* iids,
                int numEntries,
                int* transLens,
                int transLensLength,
                int* custBitmapMap,
                int** custMap,
                int* itemMap,
                SeqBitmap** f1Buff);

bool CollectBinaryInfo(
                       char* filename,
                       int& custCount,
                       int& itemCount,
                       int& transCount,
                       int*& custTransCount,
                       int*& itemCustCount,
                       int*& cids,
                       int*& tids,
                       int*& iids,
                       int*& transLens,
                       int& overallCount,
                       int& transLensLength);

bool ReadASCII(
               char* filename,
               int* cids,
               int* tids,
               int* iids,
               int numEntries,
               int* custBitmapMap,
               int** custMap,
               int* itemMap,
               SeqBitmap** f1Buff);

bool CollectASCIIInfo(
                      char* filename,
                      bool isStringFile,
                      StringMap*& custStrMap,
                      StringMap*& transStrMap,
                      StringMap*& itemStrMap,
                      int& custCount,
                      int& itemCount,
                      int& lineCount,
                      int*& custTransCount,
                      int*& itemCustCount,
                      int*& cids,
                      int*& tids,
                      int*& iids,
                      int& overallCount);

/// number of different bitmaps in SeqBitmap
const int NUM_BITMAP = 5;

/// the size of each customer data in each bitmap
const int BITMAP_LENGTH[5] =
    {
        4, 8, 16, 32, 64
    };

/// tempAndBitmap, specialBitmap, returnBitmap, SBitmap
const int NUM_BITMAPS_USED = 4;

/// Maximum length of strings used to represent
/// customers, transactions, and items when -str is used
const int MAX_STRING_SIZE = 256;

/////////////////////////////////////////////////////////////////////
/// It prints an error related to the data file input
///
/// @param errorType         the type of file read error encountered
///
/////////////////////////////////////////////////////////////////////
void PrintFileReadError(int errorType)
{
    cerr << "\nInput file error:\n\n";

    switch (errorType)
    {
    case FILEERR_64TRANSACTIONS:
        cerr << "A customer has more than 64 transactions.\n";
        break;
    case FILEERR_NOTFOUNDBINARY:
        cerr << "The input file either does not exist, or is not\n";
        cerr << "a valid binary input file (Perhaps try running\n";
        cerr << "SPAM with the -ascii flag to see if it is an\n";
        cerr << "ASCII input file)\n";
        break;
    case FILEERR_NOTFOUNDASCII:
        cerr << "The input file either does not exist, or is not\n";
        cerr << "a valid ascii input file. If the input file was\n";
        cerr << "automatically generated using a program like AssocGen,\n";
        cerr << "perhaps try running SPAM without the -ascii flag to\n";
        cerr << "see if it is a binary input file.\n";
        break;
    default:
        cerr << "Unknown file read error.\n";
    }

    cerr << "\nNotes about the input file:\n";
    cerr << "The input file should be an ASCII text file containing\n";
    cerr << "three integers, separated by spaces, on each line:\n";
    cerr << "<Customer ID> <Transaction ID> <Item ID>\n";
    cerr << "Customer IDs and Item IDs should be assigned relative to\n";
    cerr << "the overall transactional database. Transaction IDs should\n";
    cerr << "be assigned relative to the customer they belong to. Each\n";
    cerr << "customer can have no more than 64 transactions. Make sure to\n";
    cerr << "use the -ascii flag, since the input is ASCII text.\n";

    exit(0);
}

/////////////////////////////////////////////////////////////////////
/// It resizes the array to have an increased size.
///
/// @param array             pointer to the array
/// @param oldSize           the size of the array
/// @param newSize           the desired size of the array
/////////////////////////////////////////////////////////////////////
void IncArraySize(int*& array, int oldSize, int newSize)
{
    int i;

    // create a new array and copy data to the new one
    int *newArray = new int[newSize];
    for (i = 0;i < oldSize;i++)
        newArray[i] = array[i];
    for (i = oldSize;i < newSize;i++)
        newArray[i] = 0;

    // deallocate the old array and redirect the pointer to the new one
    delete [] array;
    array = newArray;
}

/////////////////////////////////////////////////////////////////////
/// It collects information about a binary data file. It finds the number of
/// customers, the number of items, the number of transactions in the dataset.
/// It also finds the number of transactions each customer has and the number
/// of customers having a particular item in their transactions.
///
/// Note:
///  - Memory should not be allocated for custTransCount and itemCustCount
///    before calling this function.
///  - Memory have to be deallocated for custTransCount and itemCustCount by
///    the caller.
///  - File format: [custID] [transID] [number of item] [itemID1, ...]
///    [custID] ...
///  - Assume that transactions of a customer appears together in the file.
///    It assures that the following case will not happen:
///   [custID-1]... [custID-1]... [custID-2]... [custID-1]...
///
/// @param filename          the filename of the data file
/// @param custCount         [output] the number of customers
/// @param itemCount         [output] the number of items
/// @param transCount        [output] the number of transactions
/// @param custTransCount    [output] number of transactions each customer has
/// @param itemCustCount     [output] number of customers having each item
///                                   in their transactions
/// @param cids	[output] customer ids exactly as they appear in the file
/// @param tids   [output] transaction ids exactly as they appear in the file
/// @param iids   [output] item ids exactly as they appear in the file
/// @param transLens [output] transaction lengths exactly as they appear in the file
/// @param overallCount   [output] length of the cids, tids, and iids arrays
/// @param transLensLength [output] length of transLens array
/// @return true - if the reading is successful.
///         false - if there is an error in the reading process.
/////////////////////////////////////////////////////////////////////
bool CollectBinaryInfo(
    char* filename,
    int& custCount,
    int& itemCount,
    int& transCount,
    int*& custTransCount,
    int*& itemCustCount,
    int*& cids,
    int*& tids,
    int*& iids,
    int*& transLens,
    int& overallCount,
    int& transLensLength)
{

    // --- Variables
    // initialize local variables
    ResizableArray * cidArr = new ResizableArray(64);
    ResizableArray * tidArr = new ResizableArray(64);
    ResizableArray * iidArr = new ResizableArray(64);
    ResizableArray * transLensArr = new ResizableArray(64);
    int custID;                   // current customer ID
    int transID;                  // current transaction ID
    int numItem;                  // number of items
    int *itemlist;                // list of items in the current transaction
    ifstream inFile;              // file handler
    int custTransSize = 400;      // size of the custTransCount array
    int itemCustSize = 400;       // size of the itemCustCount array
    bool useStdin;                // Whether or not the input is coming from stdin

    // --- Read File
    // open the binary file
    if (filename == 0)
        useStdin = true;
    else
        useStdin = false;

    if (!useStdin)
    {
        inFile.open(filename, ios::binary);
        if (!inFile.is_open())
        {
            PrintFileReadError(FILEERR_NOTFOUNDBINARY);
            return false;
        }
    }

    // initialize output variables
    // XXX- do we want these initially set to -1?
    custCount = -1;               // # of customers in the dataset (largest ID)
    itemCount = -1;               // # of items in the dataset (largest ID)
    transCount = 0;               // number of transaction
    custTransCount = new int[custTransSize];
    itemCustCount = new int[itemCustSize];

    for (int cti = 0; cti < custTransSize; cti++)
        custTransCount[cti] = 0;

    for (int ici = 0; ici < itemCustSize; ici++)
        itemCustCount[ici] = 0;

    // this array stores the ID of the previous customer we have scanned and
    // has a certain item in his/her transactions.
    int *itemPrevCustID = new int[itemCustSize];
    for (int ipi = 0; ipi < itemCustSize; ipi++)
        itemPrevCustID[ipi] = -1;

    // read data
    while ((!useStdin && !inFile.eof()) || (useStdin && !cin.eof()))
    {

        if (useStdin)
        {
            // infomation about a transaction
            cin.read((char *)&custID, sizeof(int));
            cin.read((char *)&transID, sizeof(int));
            cin.read((char *)&numItem, sizeof(int));

            itemlist = new int[numItem];

            // read in the items of the transaction
            cin.read((char *)itemlist, numItem * sizeof(int));
        }
        else
        {
            // infomation about a transaction
            inFile.read((char *)&custID, sizeof(int));
            inFile.read((char *)&transID, sizeof(int));
            inFile.read((char *)&numItem, sizeof(int));

            itemlist = new int[numItem];

            // read in the items of the transaction
            inFile.read((char *)itemlist, numItem * sizeof(int));
        }

        // just in case we reach the end of the file
        if ((!useStdin && inFile.eof()) || (useStdin && cin.eof()))
        {
            delete [] itemlist;
            break;
        }

        transLensArr->Add(numItem);
        for (int i = 0; i < numItem; i++)
        {
            // Copy the line of data into our resizable arrays
            cidArr->Add(custID);
            tidArr->Add(transID);
            iidArr->Add(itemlist[i]);
        }


        // -- update the statistcs about customers
        if (custID >= custCount)
        {
            custCount = custID + 1;

            // if custTransCount is not big enough, reallocate memory
            if (custCount > custTransSize)
            {
                int newSize = (custCount > 2 * custTransSize) ?
                              custCount : 2 * custTransSize;
                IncArraySize(custTransCount, custTransSize, newSize);
                custTransSize = newSize;
            }
        }
        custTransCount[custID]++;
        transCount++;


        // -- update the statistics about items
        for (int ici = 0; ici < numItem; ici++)
        {
            if (itemlist[ici] >= itemCount)
                itemCount = itemlist[ici] + 1;
        }

        // make sure itemCustCount and prevCust are large enough
        if (itemCount >= itemCustSize)
        {
            int newSize = (itemCount > 2 * itemCustSize) ?
                          itemCount : 2 * itemCustSize;
            IncArraySize(itemCustCount, itemCustSize, newSize);
            IncArraySize(itemPrevCustID, itemCustSize, newSize);
            itemCustSize = newSize;
        }

        for (int itemIndex = 0; itemIndex < numItem; itemIndex++)
        {
            // update itemCustCount only if the item is from a diff customer
            if (itemPrevCustID[itemlist[itemIndex]] != custID)
            {
                itemCustCount[itemlist[itemIndex]]++;
                itemPrevCustID[itemlist[itemIndex]] = custID;
            }
        }

        delete [] itemlist;
    }

    delete [] itemPrevCustID;
    if (!useStdin)
        inFile.close();

    // Copy the resizable array contents to the arrays containing
    // the in-memory cid/tid/iid lists
    cidArr->ToArray(cids, overallCount);
    tidArr->ToArray(tids, overallCount);
    iidArr->ToArray(iids, overallCount);
    transLensArr->ToArray(transLens, transLensLength);
    delete cidArr;
    delete tidArr;
    delete iidArr;
    delete transLensArr;

    return true;
}

/////////////////////////////////////////////////////////////////////
/// It collects information about an ASCII data file. It finds the number of
/// customers, the number of items, the number of line in the file. It
/// also finds the number of transactions each customer has and the number of
/// customers having a particular item in their transactions.
///
/// Note:
///  - Memory should not be allocated for custTransCount and itemCustCount
///    before calling this function.
///  - Memory have to be deallocated for custTransCount and itemCustCount by
///    the caller.
///  - File format: [custID] [transID] [itemID]
///        [custID] [transID] [itemID] ...
///  - Assume that transactions of a customer appears together in the file.
///    It assures that the following case will not happen:
///    [custID-1]... [custID-1]... [custID-2]... [custID-1]...
///  - Assume that items of a transactions appears together in the file.
///    It assures that the following case will not happen:
///   ... [transID-1] ...
///   ... [transID-2] ...
///   ... [transID-1] ...
///
/// @param filename          the filename of the data file
/// @param isStringFile      whether the input file contains integers or the string names
/// @param custStrMap        [output] Maps cust IDs to strings (only used when isStringFile == true)
/// @param transStrMap       [output] Maps trans IDs to strings (only used when isStringFile == true)
/// @param itemStrMap        [output] Maps item IDs to strings (only used when isStringFile == true)
/// @param custCount         [output] the number of customers
/// @param itemCount         [output] the number of items
/// @param lineCount         [output] the number of line in the file
/// @param custTransCount    [output] number of transactions each customer has
/// @param itemCustCount     [output] number of customers having each item in
///                                   their transactions
/// @param cids	[output] customer ids exactly as they appear in the file
/// @param tids   [output] transaction ids exactly as they appear in the file
/// @param iids   [output] item ids exactly as they appear in the file
/// @param overallCount   [output] length of the cids, tids, and iids arrays
///
/// @return true - if the reading is successful.
///         false - if there is an error in the reading process
/////////////////////////////////////////////////////////////////////
bool CollectASCIIInfo(
    char* filename,
    bool isStringFile,
    StringMap*& custStrMap,
    StringMap*& transStrMap,
    StringMap*& itemStrMap,
    int& custCount,
    int& itemCount,
    int& lineCount,
    int*& custTransCount,
    int*& itemCustCount,
    int*& cids,
    int*& tids,
    int*& iids,
    int& overallCount)
{

    // --- Variables
    // initialize local variables
    ResizableArray * cidArr = new ResizableArray(64);
    ResizableArray * tidArr = new ResizableArray(64);
    ResizableArray * iidArr = new ResizableArray(64);
    int custID;                   // current customer ID
    int transID;                  // current transaction ID
    int itemID;                   // current item ID
    int prevTransID = -1;         // previous transaction ID
    ifstream inFile;              // file handler
    int custTransSize = 400;      // size of the custTransCount array
    int itemCustSize = 400;       // size of the itemCustCount array
    int i;                        // counter
    bool useStdin;                // whether or not the input will come from stdin
    int custStrMapID = 1;
    int transStrMapID = 1;
    int itemStrMapID = 1;

    // If we are mapping strings to IDs, initialize the StringMaps
    if (isStringFile)
    {
        custStrMap = new StringMap();
        transStrMap = new StringMap();
        itemStrMap = new StringMap();
    }

    // --- Read File
    // open the ASCII file; if filename is null then use stdin
    if (filename == 0)
        useStdin = true;
    else
        useStdin = false;

    if (!useStdin)
    {
        inFile.open(filename);
        if (!inFile.is_open())
        {
            PrintFileReadError(FILEERR_NOTFOUNDASCII);
            return false;
        }
    }

    // initialize output variables
    custCount = -1;               // # of customers in the dataset (largest ID)
    itemCount = -1;               // # of items in the dataset (largest ID)
    lineCount = 0;                // number of transaction
    custTransCount = new int[custTransSize];
    itemCustCount = new int[itemCustSize];
    for (i = 0; i < custTransSize; i++)
        custTransCount[i] = 0;
    for (i = 0; i < itemCustSize; i++)
        itemCustCount[i] = 0;


    // this array stores the ID of the previous customer we have scanned and
    // has a certain item in his/her transactions.
    int *itemPrevCustID = new int[itemCustSize];
    for (i = 0; i < itemCustSize; i++)
        itemPrevCustID[i] = -1;

    // read data
    while ((!useStdin && !inFile.eof()) || (useStdin && !cin.eof()))
    {
        // read in the transaction
        if (isStringFile)
        {
            // Read in the 3 strings corresponding to customer name,
            // transaction name (will probably be an int but we will map it to a string anyway)
            // and item name.
            char *custStr = new char[MAX_STRING_SIZE];
            char *transStr = new char[MAX_STRING_SIZE];
            char *itemStr = new char[MAX_STRING_SIZE];
            if (useStdin)
            {
                cin.getline(custStr, MAX_STRING_SIZE);
                cin.getline(transStr, MAX_STRING_SIZE);
                cin.getline(itemStr, MAX_STRING_SIZE);
            }
            else
            {
                inFile.getline(custStr, MAX_STRING_SIZE);
                inFile.getline(transStr, MAX_STRING_SIZE);
                inFile.getline(itemStr, MAX_STRING_SIZE);
            }

            // If each string was found in our map, reuse its ID
            // Otherwise assign it a new ID
            const int * custKeyID = custStrMap->GetKey(custStr);
            const int * transKeyID = transStrMap->GetKey(transStr);
            const int * itemKeyID = itemStrMap->GetKey(itemStr);
            if (custKeyID != 0)
                custID = *custKeyID;
            else
            {
                custID = custStrMapID;
                custStrMap->Add(custID, custStr);
                custStrMapID++;
            }
            if (transKeyID != 0)
                transID = *transKeyID;
            else
            {
                transID = transStrMapID;
                transStrMap->Add(transID, transStr);
                transStrMapID++;
            }
            if (itemKeyID != 0)
                itemID = *itemKeyID;
            else
            {
                itemID = itemStrMapID;
                itemStrMap->Add(itemID, itemStr);
                itemStrMapID++;
            }
        }
        else
        {
            if (useStdin)
            {
                cin >> custID;
                cin >> transID;
                cin >> itemID;
            }
            else
            {
                inFile >> custID;
                inFile >> transID;
                inFile >> itemID;
            }
        }

        // Copy the line of data into our resizable arrays
        cidArr->Add(custID);
        tidArr->Add(transID);
        iidArr->Add(itemID);

        // -- update the statistcs about customers
        if (custID >= custCount)
        {
            custCount = custID + 1;

            // make sure custTransCount is big enough
            if (custCount > custTransSize)
            {
                int newSize = (custCount > 2 * custTransSize) ?
                              custCount : 2 * custTransSize;
                IncArraySize(custTransCount, custTransSize, newSize);
                custTransSize = newSize;
            }
			prevTransID = -1;
        }

        // increment custTransCount only if it's a different transaction
        if (prevTransID != transID)
        {
            custTransCount[custID]++;
            prevTransID = transID;
        }
        lineCount++;

        // -- update the statistics about items
        if (itemID >= itemCount)
        {
            itemCount = itemID + 1;

            // make sure itemCustCount is large enough
            if (itemCount >= itemCustSize)
            {
                int newSize = (itemCount > 2 * itemCustSize) ?
                              itemCount : 2 * itemCustSize;
                IncArraySize(itemCustCount, itemCustSize, newSize);
                IncArraySize(itemPrevCustID, itemCustSize, newSize);
                itemCustSize = newSize;
            }
        }

        // update itemCustCount only if the item is from a different customer
        if (itemPrevCustID[itemID] != custID)
        {
            itemCustCount[itemID]++;
            itemPrevCustID[itemID] = custID;
        }
    }

    delete [] itemPrevCustID;
    if (!useStdin)
        inFile.close();

    // Copy the resizable array contents to the arrays containing
    // the in-memory cid/tid/iid lists
    cidArr->ToArray(cids, overallCount);
    tidArr->ToArray(tids, overallCount);
    iidArr->ToArray(iids, overallCount);
    delete cidArr;
    delete tidArr;
    delete iidArr;

    return true;
}

/////////////////////////////////////////////////////////////////////
/// It reads stores the binary data file.
///
/// Note:
///  - File format: [custID] [transID] [number of item] [itemID1, ...]
///    [custID] ...
///  - Assume that transactions of a customer appears together in the file.
///    It assures that the following case will not happen:
///    [custID-1]... [custID-1]... [custID-2]... [custID-1]...
///
/// @param filename          the filename of the data file
/// @param cids              the list of customer IDs as it appears in the file
/// @param tids              the list of transaction IDs
/// @param iids              the list of item IDs
/// @param numEntries        the length of the arrays cids, tids, and iids
/// @param transLens         the list of transaction lengths
/// @param transLensLength   the length of the list transLens
/// @param custBitmapMap     map from custID to bitmapID
/// @param custMap           the mapping of custID from external naming
///                              to internal naming
/// @param itemMap           the mapping of itemID from external naming
///                              to internal naming
/// @param f1Buff            [output] buffer for frequent-one item sets
///
/// @return true - if the reading is successful.
///         false - if there is an error in the reading process
/////////////////////////////////////////////////////////////////////
bool ReadBinary(
    char* filename,
    int* cids,
    int* tids,
    int* iids,
    int numEntries,
    int* transLens,
    int transLensLength,
    int* custBitmapMap,
    int** custMap,
    int* itemMap,
    SeqBitmap** f1Buff)
{

    // --- Variables
    // initialize local variables
    int custID;              // current customer ID
    int transID;             // current transaction ID
    int numItem;             // number of items
    int *itemlist;           // list of items in the current transaction
    int prevCustID = -1;
    int bitmapID = 0;
    int index = 0;
    ifstream inFile;         // file handler

    // --- Read File
    // Always do a second scan for binary files
    bool secondScan = false;
    int lenIndex = 0;
    int scanIndex = 0;

    if (secondScan)
    {
        if (filename == 0)
        {
            cout << "Error: cannot read input a second time when -stdin is on";
            exit(-1);
        }

        // open the binary file
        inFile.open(filename, ios::binary);
        if (!inFile.is_open())
        {
            return false;
        }
    }

    while (     (secondScan && !inFile.eof())
                || (!secondScan && scanIndex < numEntries) )
    {
        if (secondScan)
        {
            inFile.read((char *)&custID, sizeof(int));
            inFile.read((char *)&transID, sizeof(int));
            inFile.read((char *)&numItem, sizeof(int));
            itemlist = new int[numItem];

            // read in the items of the transaction
            inFile.read((char *)itemlist, numItem * sizeof(int));

            // just in case
            if (inFile.eof())
                break;
        }
        else
        {
            numItem = transLens[lenIndex];
            itemlist = new int[numItem];
            custID = cids[scanIndex];
            transID = tids[scanIndex];
            for (int i = 0; i < numItem; i++)
                itemlist[i] = iids[scanIndex + i];
            scanIndex+=numItem;
            lenIndex++;
        }

        if (custID != prevCustID)
        {
            prevCustID = custID;
            bitmapID = custBitmapMap[custID];
            index = custMap[bitmapID][custID] * BITMAP_LENGTH[bitmapID];
        }

        // Fill in transaction bit for appropriate bitmaps
        for (int j = 0; j < numItem; j++)
        {
            if (itemMap[itemlist[j]] >= 0)
                f1Buff[itemMap[itemlist[j]]]->
                FillEmptyPosition(bitmapID, index);
        }


        index++;
        delete [] itemlist;
    }

    if (secondScan)
        inFile.close();
    return true;
}

/////////////////////////////////////////////////////////////////////
/// It reads stores the ASCII data file.
///
/// Note:
///  - File format: [custID] [transID] [itemID]
///        [custID] [transID] [itemID]
///        ...
///  - Assume that transactions of a customer appears together in the file.
///    It assures that the following case will not happen:
///   [custID-1]... [custID-1]... [custID-2]... [custID-1]...
///  - Assume that items of a transactions appears together in the file.
///    It assures that the following case will not happen:
///   ... [transID-1] ...
///   ... [transID-2] ...
///   ... [transID-1] ...
///
/// @param filename          the filename of the data file
/// @param cids              the list of customer IDs as it appears in the file
/// @param tids              the list of transaction IDs
/// @param iids              the list of item IDs
/// @param numEntries        the length of the arrays cids, tids, and iids
/// @param custBitmapMap     map from custID to bitmapID
/// @param custMap           the mapping of custID from external naming
///                              to internal naming
/// @param itemMap           the mapping of itemID from external naming
///                              to internal naming
/// @param f1Buff            [output] buffer for frequent-one item sets
///
/// @return true - if the reading is successful.
///         false - if there is an error in the reading process
/////////////////////////////////////////////////////////////////////
bool ReadASCII(
    char* filename,
    int* cids,
    int* tids,
    int* iids,
    int numEntries,
    int* custBitmapMap,
    int** custMap,
    int* itemMap,
    SeqBitmap** f1Buff)
{

    // --- Variables
    // initialize local variables
    int custID;              // current customer ID
    int transID;             // current transaction ID
    int itemID;              // current item ID
    int prevTransID = -1;    // previous transaction ID
    int prevCustID = -1;     // previous customer ID
    int bitmapID = 0;        // bitmap used to store current customer data
    int index = 0;
    ifstream inFile;         // file handler

    // --- Read File
    bool secondScan = false;
    int scanIndex = 0;

    if (secondScan)
    {
        if (filename == 0)
        {
            cout << "Error: cannot read input a second time when -stdin is on";
            exit(-1);
        }
        // open the ASCII file
        inFile.open(filename);
        if (!inFile.is_open())
        {
            return false;
        }
    }

    // read and store data
    while (   (secondScan && !inFile.eof())
              || (!secondScan && scanIndex < numEntries))
    {

        if (secondScan)
        {
            // read in the transaction
            inFile >> custID;

            if (inFile.eof())
                break;

            inFile >> transID;
            inFile >> itemID;
        }
        else
        {
            custID = cids[scanIndex];
            transID = tids[scanIndex];
            itemID = iids[scanIndex];
            scanIndex++;
        }

        if (custID != prevCustID)
        {
            prevCustID = custID;
            bitmapID = custBitmapMap[custID];
            index = custMap[bitmapID][custID] * BITMAP_LENGTH[bitmapID] - 1;
			prevTransID = -1;
		}
		
		if (prevTransID != transID)
        {
            index++;
            prevTransID = transID;
        }

        // Fill in transaction bit for appropriate bitmaps
        if (itemMap[itemID] >= 0)
            f1Buff[itemMap[itemID]]->FillEmptyPosition(bitmapID, index);
    }

    if (secondScan)
        inFile.close();

    return true;
}

/////////////////////////////////////////////////////////////////////
/// It reads the input file and finds the frequent-1 itemsets.
///
/// @param isBinaryFile      whether the input file is a binary data file
/// @param isStringFile      whether the input file contains integers or the string names
/// @param filename          the filename of the data file
/// @param minSupPercent     the minimum support percentage
/// @param custStrMap        [output] Maps cust IDs to strings (only used when isStringFile == true)
/// @param transStrMap       [output] Maps trans IDs to strings (only used when isStringFile == true)
/// @param itemStrMap        [output] Maps item IDs to strings (only used when isStringFile == true)
///
/// @return DatasetInfo - the information gathered from the dataset
/////////////////////////////////////////////////////////////////////
DatasetInfo* ReadDataset(
    bool isBinaryFile,
    bool isStringFile,
    char *filename,
    double minSupPercent,
    StringMap *&custStrMap,
    StringMap *&transStrMap,
    StringMap *&itemStrMap)
{

    DatasetInfo* info = new DatasetInfo(); // dataset info

    // for the first pass of the dataset
    int tempCustCount = -1;  // # of customers in the dataset (largest ID)
    int itemCount = -1;      // # of items in the dataset (largest ID)
    int transCount = 0;      // number of transaction
    int *custTransCount;     // number of transactions each customer has
    int *itemCustCount;      // number of customers having that item in
    //     their transactions

    // for post-processing of the info of the dataset
    int *bitmapSizes;        // number of customer in each bitmap
    //     (4, 8, 16, 32, and 32+)
    int *custBitmapMap;      // map customers to bitmapIDs
    int **custMap;           // map bitmapID and custID to index

    int noTransCount = 0;    // the number of customers with no transaction
    int *itemMap;            // items renaming information

    int numCompression;      // maximum number of compression that we will do

    bool result = false;     // true if successfully read in data
    int i, j;                // temp variables

    // ----------------------- First scan of the data ----------------------//
    // This is the first scan of the database. In this scan, we gather
    // information about the dataset, for example, the number of items, the
    // number of customers, and the number of transaction for each customers.
    // ---------------------------------------------------------------------//

    // Copy the file's contents into memory so we can avoid a second file scan
    int *cids;
    int *tids;
    int *iids;
    int numEntries;
    int *transLens;
    int transLensLength;

    if (isBinaryFile)
    {
        result = CollectBinaryInfo(
                     filename,
                     tempCustCount,
                     itemCount,
                     transCount,
                     custTransCount,
                     itemCustCount,
                     cids,
                     tids,
                     iids,
                     transLens,
                     numEntries,
                     transLensLength);
    }
    else
    {
        result = CollectASCIIInfo(
                     filename,
                     isStringFile,
                     custStrMap,
                     transStrMap,
                     itemStrMap,
                     tempCustCount,
                     itemCount,
                     transCount,
                     custTransCount,
                     itemCustCount,
                     cids,
                     tids,
                     iids,
                     numEntries);
    }
    if (!result)
    {
        delete info;
        return 0;
    }

    // ----------------------- First scan of the data ----------------------//
    // We process the data so that we can effectively store the data.
    // ----------------------- First scan of the data ----------------------//

    // *-- customer info --*
    // This section:
    // - finds out the max number of transactions and number of customers
    //   who have transactions in the dataset.
    // - creates a mapping which maps ids of customers who have transations
    //   in the dataset to bitmapIDs. Currently, we use five bitmap sizes:
    //   4, 8, 16, 32, and 64.
    // - finds the size of each bitmaps
    // - given bitmapIDs of customers, creates a mapping which maps ids of
    //   customers and bitmapIDs to a set of integers, which is the index of
    //   the customer in each bitmap.


    // initialize variables
    info->maxCustTrans = 0;
    noTransCount = 0;
    custBitmapMap = new int[tempCustCount]; // custID --> bitmapID
    bitmapSizes = new int[NUM_BITMAP];      // bitmap size
    custMap = new int * [NUM_BITMAP];       // bitmapID, custID -->
    //     index in bitmapID
    for (i = 0; i < NUM_BITMAP; i++)
    {
        custMap[i] = new int[tempCustCount];
        bitmapSizes[i] = 0;
    }

    // create the maps
    for (i = 0; i < tempCustCount; i++)
    {
        if (custTransCount[i] > BITMAP_LENGTH[NUM_BITMAP - 1])
        {
            // Spam does not support customers that have more than 64
            // transactions; give the user an error message.
            PrintFileReadError(FILEERR_64TRANSACTIONS);
        }

        if (custTransCount[i] <= 0)
        {
            // this customer doesn't have transactions, we won't consider it
            custBitmapMap[i] = -1;
            noTransCount++;
        }
        else
        {
            // find out which bitmap this customer fit in (find the bitmapID)
            for (j = 0; j < NUM_BITMAP; j++)
                if (custTransCount[i] <= BITMAP_LENGTH[j])
                {
                    custBitmapMap[i] = j;
                    break;
                }
        }

        // create the mapping from bitmapID & custID to index in the bitmap
        // with bitmapID
        for (j = 0; j < NUM_BITMAP; j++)
            if (custBitmapMap[i] == j)
            {
                custMap[j][i] = bitmapSizes[j];
                bitmapSizes[j]++;
            }
            else
                custMap[j][i] = -1;

        // find the max number of transactions
        if (custTransCount[i] > info->maxCustTrans && custTransCount[i] <= 64)
            info->maxCustTrans = custTransCount[i];
    }

    // actual number of customers
    info->custCount = tempCustCount - noTransCount;
    info->bitmapSizes = new int[NUM_BITMAP];
    for (i = 0; i < NUM_BITMAP; i++)
        info->bitmapSizes[i] = bitmapSizes[i];

    // *-- minimum support --*
    // user-specified minimum support; if the minSup is not greater than 0,
    // we assume it to be 1

#ifdef ___PREFIXSPAN___

    info->minSup = (int) ceil(minSupPercent * info->custCount);
#else

    info->minSup = (int) floor(minSupPercent * info->custCount + 0.5);
#endif

    info->minSup = (info->minSup == 0) ? 1 : info->minSup;

    // *-- Frequent 1 itemsets --*
    // This section:
    // - finds the frequent 1 itemset
    // - creates a mapping which maps item ids of the frequent 1 itemset
    //   to a set of consecutive integers

    info->f1Size = 0;
    itemMap = new int[itemCount];

    for (i = 0; i < itemCount; i++)
    {
        if (itemCustCount[i] >= info->minSup )
        {
            itemMap[i] = info->f1Size;
            info->f1Size++;
        }
        else
        {
            itemMap[i] = -1;
        }
    }

    // return if there is no frequent 1 itemset
    if (info->f1Size == 0)
        return info;

    numCompression = info->maxCustTrans * info->f1Size;

    // Allocate the SeqBitmap buffer
    // -- estimate the max size of the bitmap data buffers
    int size64 = bitmapSizes[4];
    int size32 = bitmapSizes[3] + size64;
    int size16 = bitmapSizes[2] + size32;
    int size8 = bitmapSizes[1] + size16;
    int size4 = bitmapSizes[0] + size8;

    size4 = Bitmap4::CalcSize(size4);
    size8 = Bitmap8::CalcSize(size8);
    size16 = Bitmap16::CalcSize(size16);
    size32 = Bitmap32::CalcSize(size32);
    size64 = Bitmap64::CalcSize(size64);


    // Size of bitmap * number of levels * number of bitmaps per level
    SeqBitmap::MemAlloc(size4 * (info->maxCustTrans * info->f1Size) *
                        (NUM_BITMAPS_USED + info->f1BufSize),
                        size8 * (info->maxCustTrans * info->f1Size) *
                        (NUM_BITMAPS_USED + info->f1BufSize),
                        size16 * (info->maxCustTrans * info->f1Size) *
                        (NUM_BITMAPS_USED + info->f1BufSize),
                        size32 * (info->maxCustTrans * info->f1Size) *
                        (NUM_BITMAPS_USED + info->f1BufSize),
                        size64 * (info->maxCustTrans * info->f1Size) *
                        (NUM_BITMAPS_USED + info->f1BufSize));


    // -- estimate the max size of the f1 buffers
    // size of f1 buffer depends on num of compression that we allow
    // for f1Buff and f1NameBuff
    info->f1BufSize = info->f1Size * (numCompression + 1) + 10;

    info->f1Buff = new SeqBitmap * [info->f1BufSize];

    for (i = 0; i < info->f1Size; i++)
        info->f1Buff[i] = new SeqBitmap(
                              bitmapSizes[0],
                              bitmapSizes[1],
                              bitmapSizes[2],
                              bitmapSizes[3],
                              bitmapSizes[4]);

    //cout << bitmapSizes[0] << " " << bitmapSizes[1] << " "
    // << bitmapSizes[2] << " " << bitmapSizes[3] << " "
    // << bitmapSizes[4] << endl;
    for (; i < info->f1BufSize; i++)
        info->f1Buff[i] = 0;

    info->f1NameBuff = new int[info->f1BufSize];
    for (i = 0; i < itemCount; i++)
        if (itemMap[i] >= 0)
            info->f1NameBuff[itemMap[i]] = i;

    // -- estimate the max size of slist buffers
    // Size of sListBuff depends on the max number of transactions
    // a customer has. In the worst case, we will only go down "s-tree"
    // for info->maxCustTrans number of times, and we only change s-list
    // when we go down "s-tree".
    info->sListBuff = new int[(info->maxCustTrans * info->f1Size)
                              * info->f1Size + 1];

    // -- estimate the max size of ilist buffers
    // Size of i-list buff also depends on the max number of transactions
    // a customer has. It aslo depends on how many times it goes down the
    // "i-tree". In the worst case we will go down the whole "i-tree" for
    // each transaction in the sequence.
    info->iListBuff = new int[(info->maxCustTrans * info->f1Size )
                              * info->f1Size + 1];

    for (i = 0; i < info->f1Size; i++)
        info->sListBuff[i] = i;

    // -- estimate the max size of count buffer
    info->countBuff = new CountInfo[info->f1Size];




    // ---------------------- Second scan of the data ----------------------//
    // This is the second scan of the database. In this scan, we actually
    // retrieve the information of each transaction. We will also partition
    // the data by customers.
    // ---------------------------------------------------------------------//

    if (isBinaryFile)
        result = ReadBinary(
                     filename,
                     cids,
                     tids,
                     iids,
                     numEntries,
                     transLens,
                     transLensLength,
                     custBitmapMap,
                     custMap,
                     itemMap,
                     info->f1Buff);
    else
        result = ReadASCII(
                     filename,
                     cids,
                     tids,
                     iids,
                     numEntries,
                     custBitmapMap,
                     custMap,
                     itemMap,
                     info->f1Buff);

    if (!result)
    {
        delete info;
        return 0;
    }

    // deallocate memory storing file to avoid second scan
    delete [] cids;
    delete [] tids;
    delete [] iids;

    // deallocate memory for counting customer index
    delete [] custTransCount;
    delete [] itemCustCount;
    delete [] custBitmapMap;
    for (i = 0; i < NUM_BITMAP; i++)
        delete [] custMap[i];
    delete [] custMap;
    delete [] bitmapSizes;
    delete [] itemMap;

    return info;
}

/** @} */