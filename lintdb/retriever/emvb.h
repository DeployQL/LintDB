#pragma once

/**
 * AVX2 isn't quite ready yet. I'm not the biggest expert in intrinsics, and we're looking to adjust from AVX512 instructions to AVX2.
 * For now, we'll only use the generic library until emvb_avx.h is ready.
*/
// #ifdef __AVX2__
// #include "lintdb/retriever/emvb_avx.h"
// #else
// #endif

#include "lintdb/retriever/emvb_generic.h"
