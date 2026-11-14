#pragma once
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <cstring>

using half = c10::Half;
#define __mlu_entry__
#define __mlu_func__
#define __nram__
#define __wram__

// A single emulated task executes the same strided loops as all device tasks.
constexpr int taskId = 0;
constexpr int taskDim = 1;
