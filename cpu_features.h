#pragma once
#include <immintrin.h>

class CPUFeatures {
public:
    static bool init() {
        if (!initialized) {
            #if defined(_MSC_VER)
                int cpuInfo[4];
                __cpuid(cpuInfo, 1);
                has_sse2 = cpuInfo[3] & (1 << 26);
                has_avx2 = __builtin_cpu_supports("avx2");
            #else
                has_sse2 = __builtin_cpu_supports("sse2");
                has_avx2 = __builtin_cpu_supports("avx2");
            #endif
            initialized = true;
        }
        return true;
    }

    static bool hasSSE2() { return init() && has_sse2; }
    static bool hasAVX2() { return init() && has_avx2; }
    

    static bool initialized;
    static bool has_sse2;
    static bool has_avx2;
};