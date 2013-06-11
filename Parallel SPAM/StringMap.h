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
#include <memory.h>
#include <string.h>

/////////////////////////////////////////////////////////////////////
///
/// StringMap.h
///
/////////////////////////////////////////////////////////////////////
#ifndef __StringMap_H__
#define __StringMap_H__

/////////////////////////////////////////////////////////////////////
/// @addtogroup GlobalFunctions
/** @{ */

/////////////////////////////////////////////////////////////////////
/// A data structure that acts as a two-way one-to-one mapping from integers to strings
/// Optimized for SPAM to quickly retrieve customer IDs, transaction IDs,
/// and item IDs from their corresponding string names, and vice versa
/////////////////////////////////////////////////////////////////////
class StringMap
{
public:
    /////////////////////////////////////////////////////////////////////
    /// Allocate the memory for the default initial size of the array
    ///
    /////////////////////////////////////////////////////////////////////
    StringMap()
    {
        _keyMem = new int[DEFAULT_INITIAL_SIZE];
        _valueMem = new const char*[DEFAULT_INITIAL_SIZE];
        memSize = DEFAULT_INITIAL_SIZE;
        length = 0;
    };

    /////////////////////////////////////////////////////////////////////
    /// Destructor
    /////////////////////////////////////////////////////////////////////
    ~StringMap()
    {
        delete [] _keyMem;
        delete [] _valueMem;
    };

    /////////////////////////////////////////////////////////////////////
    /// Add an item to the StringMap
    ///
    /// @param key The integer key of this item
    /// @param value The string value of this item
    /////////////////////////////////////////////////////////////////////
    void Add(int key, const char* value)
    {
        if (length < memSize)
        {
            _keyMem[length] = key;
            _valueMem[length] = value;
        }
        else
        {
            // Double the size of our local array

            // Allocate temp arrays with double the size
            int * newKeyMem = new int[memSize*2];
            const char ** newValueMem = new const char*[memSize*2];
            memcpy(newKeyMem, _keyMem, sizeof(int)*memSize);
            memcpy(newValueMem, _valueMem, sizeof(const char *)*memSize);
            delete [] _keyMem;
            delete [] _valueMem;

            // Set our memory arrays to the temp arrays
            _keyMem = newKeyMem;
            _valueMem = newValueMem;
            _keyMem[length] = key;
            _valueMem[length] = value;
            memSize = memSize*2;
        }
        length++;
    };

    /////////////////////////////////////////////////////////////////////
    /// Return the value associated with a given key, or null if the key/value pair
    /// is not in the StringMap
    ///
    /// @param key The key to look up
    /// @return The value associated with this key, or null if the key/value pair is not in the StringMap
    /////////////////////////////////////////////////////////////////////
    const char * GetValue(int key)
    {
        for (int i = 0; i < length; ++i)
        {
            if (_keyMem[i] == key)
                return _valueMem[i];
        }
        return 0;
    }

    /////////////////////////////////////////////////////////////////////
    /// Return the key associated with a given value, or null if the key/value pair
    /// is not in the StringMap
    ///
    /// @param value The value to look up
    /// @return The key associated with this value, or null if the key/value pair is not in the StringMap
    /////////////////////////////////////////////////////////////////////
    const int * GetKey(const char* value)
    {
        for (int i = 0; i < length; ++i)
        {
            if (strcmp(_valueMem[i],value) == 0)
            {
                const int * retVal = new int(_keyMem[i]);
                return retVal;
            }
        }
        return 0;
    }


private:
    static const int DEFAULT_INITIAL_SIZE = 16;

    // --- private variables
    int * _keyMem;	            // The internal memory storing the keys
    const char ** _valueMem;    // The internal memory storing the strings
    int memSize;		        // The number of key/value pairs we have allocated memory for
    int length;			        // The number of key/value pairs the user has added
}
; // class StringMap

/** @} */
#endif

