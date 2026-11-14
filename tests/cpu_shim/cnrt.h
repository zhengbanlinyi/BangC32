#pragma once
#include <cstdint>
using cnrtQueue_t = void *;
using cnrtFunctionType_t = int;
struct cnrtDim3_t { unsigned int x, y, z; };
constexpr int cnrtFuncTypeBlock = 0;
constexpr int cnrtSuccess = 0;
inline int cnrtGetDevice(int *device) { *device = 0; return cnrtSuccess; }
