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
/// TreeNode.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __TREE_H__
#define __TREE_H__

#include "SeqBitmap.h"
#include "DatasetInfo.h"

/////////////////////////////////////////////////////////////////////
/// @ingroup AlgorithmicComponents
///
/// A class for representing nodes in the search tree.
/////////////////////////////////////////////////////////////////////
class TreeNode
{

public:
    /// Constructor
    TreeNode()
    {
        f1 = 0;
        f1Name = 0;
        f1Size = 0;
        sList = 0;
        sLength = 0;
        iList = 0;
        iLength = 0;
        iBitmap = 0;
        level = 0;
    }

    /// Destructor
    ~TreeNode()
    {}


    /// bitmaps of frequent-1 itemsets
    SeqBitmap** f1;

    /// names of the corresponding frequent-1 itemsets
    int* f1Name;

    /// number of frequent-1 itemsets
    int f1Size;

    /// list of s-extensions for this node
    int* sList;

    /// number of s-extensions
    int sLength;

    /// list of i-extensions for this node
    int* iList;

    /// number of i-extensions
    int iLength;

    /// the i-bitmap of the current node
    SeqBitmap* iBitmap;

    /// overall level of this node in the tree
    int level;

    /// the number of s-steps that have been taken to reach this node in the tree
    int sLevel;

    /// Whether or not compression will be performed at this node
    bool compress;

    /// Support count information for all of the frequent extensions from this node
    static CountInfo* countList;

};

#endif
