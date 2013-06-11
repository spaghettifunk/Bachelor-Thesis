// VS2010 - directives for avoiding the red underline on __syncthreads()
#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif
// ------------------------------------------------------------------------