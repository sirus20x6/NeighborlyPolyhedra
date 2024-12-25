#pragma once
#include <immintrin.h>
#include <random>
#include <Eigen/Dense>
#include "plane.h"
#include "util.h"
#include <iostream>

extern bool g_debug_mode;
namespace simd {

float pt_to_line_dist_sq_simd(const Vector2f& pt, const Vector2f& a, const Vector2f& b);
void add_random_noise_simd(const float* input, float sigma, float* output, size_t size);
void plane_intersection_simd(Vector3f& result,
                           const Vector3f& n1, float d1,
                           const Vector3f& n2, float d2,
                           const Vector3f& n3, float d3);
int count_crossings_simd(const Verts3D& v3ds, const Plane& plane, const Face& poly);

inline void debugLog(const std::string& msg) {
    if (g_debug_mode) std::cout << msg;
}
template<typename T>
inline std::string vec_to_string(const T& v) {
    std::stringstream ss;
    ss << v.transpose();
    return ss.str();
}
inline __m128 box_muller_transform(std::mt19937& gen, std::uniform_real_distribution<float>& dist);

} // namespace simd