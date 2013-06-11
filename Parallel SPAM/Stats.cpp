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
/// Stats.cpp
/////////////////////////////////////////////////////////////////////

#include "Stats.h"

using namespace std;

Stats::Stats(int n)
{
    numLevels = n;
    levelTime = new double[numLevels];
    compTime = new double[numLevels];
    s_ext = new int[numLevels];
    i_ext = new int[numLevels];
    numNodes = new int[numLevels];
    numComp = new int[numLevels];

    for (int i = 0; i < numLevels; i++)
    {
        levelTime[i] = 0.0;
        compTime[i] = 0.0;
        s_ext[i] = 0;
        i_ext[i] = 0;
        numNodes[i] = 0;
        numComp[i] = 0;
    }
}

Stats::~Stats()
{
    delete [] levelTime;
    delete [] compTime;
    delete [] s_ext;
    delete [] i_ext;
    delete [] numNodes;
    delete [] numComp;
}

void Stats::startLevelTimer()
{
    levelStartTime = clock();
}

void Stats::stopLevelTimer(int level)
{
    int levelStopTime = clock();
    levelTime[level] += (levelStopTime - levelStartTime) /
                        (double)CLOCKS_PER_SEC;
}

void Stats::startCompTimer()
{
    compStartTime = clock();
}

void Stats::stopCompTimer(int level)
{
    int compStopTime = clock();
    compTime[level] += (compStopTime - compStartTime) /
                       (double)CLOCKS_PER_SEC;
}

void Stats::updateSExt(int level, int s)
{
    s_ext[level] += s;
}

void Stats::updateIExt(int level, int i)
{
    i_ext[level] += i;
}

void Stats::updateNumNodes(int level, int n)
{
    numNodes[level] += n;
}

void Stats::updateNumComp(int level, int c)
{
    numComp[level] += c;
}

void Stats::printStats(ofstream &output)
{
    for (int i = 0; i < numLevels; i++)
    {
        if (numNodes[i] == 0)
        {
            break;
        }

        output << "Level " << i << endl;
        output << "\t Num nodes: " << numNodes[i] << endl;
        output << "\t Num comp: " << numComp[i] << endl;
        output << "\t Total level time: " << levelTime[i] << endl;
        output << "\t Total comp time: " << compTime[i] << endl;
        output << "\t Average level time: "
        << (levelTime[i] / (double)numNodes[i]) << endl;
        output << "\t Average comp time: "
        << (compTime[i] / (double)numNodes[i]) << endl;
        output << "\t Average s ext: "
        << (s_ext[i] / (double)numNodes[i]) << endl;
        output << "\t Average i ext: "
        << (i_ext[i] / (double)numNodes[i]) << endl << endl;
    }
}
