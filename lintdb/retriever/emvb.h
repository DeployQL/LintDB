#pragma once

#ifdef __AVX2__
#include "lintdb/retriever/emvb_avx.h"
#else
#include "lintdb/retriever/emvb_generic.h"
#endif