#include "simd_util.h"
#include "cpu_features.h"
#include <cmath>
#include <iostream>

bool g_debug_mode = false;

// Helper function to access __m128 values since m128_f32 isn't available
inline float get_float(__m128 v, int index) {
    alignas(16) float temp[4];
    _mm_store_ps(temp, v);
    return temp[index];
}

namespace simd {

inline __m128 box_muller_transform(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    alignas(16) float u[4], v[4];
    
    // Generate 4 pairs of uniform random numbers
    for (int i = 0; i < 4; ++i) {
        u[i] = dist(gen);
        v[i] = dist(gen);
    }

    // Load into SSE registers
    __m128 u_vec = _mm_load_ps(u);
    __m128 v_vec = _mm_load_ps(v);
    
    alignas(16) float results[4];
    
    // For each pair, calculate Box-Muller transform using scalar math
    // since we don't have direct SIMD transcendental functions
    for (int i = 0; i < 4; ++i) {
        float r = std::sqrt(-2.0f * std::log(u[i]));
        float theta = 2.0f * 3.14159265359f * v[i];
        results[i] = r * std::cos(theta);
    }
    
    return _mm_load_ps(results);
}

float pt_to_line_dist_sq_simd(const Vector2f& pt, const Vector2f& a, const Vector2f& b) {
    // Use exactly the same formula as scalar version
    const Vector2f p = pt - a;
    const Vector2f d = b - a;
    const float t = p.dot(d) / d.squaredNorm();
    const Vector2f v = p - d * std::min(std::max(t, 0.0f), 1.0f);
    return v.squaredNorm();
}

void add_random_noise_simd(const float* input, float sigma, float* output, size_t size) {
    if (!CPUFeatures::hasAVX2()) {
        throw std::runtime_error("AVX2 required for SIMD functions");
    }
    
    const __m128 sigma_vec = _mm_set1_ps(sigma);
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        __m128 noise = box_muller_transform(gen, dist);
        noise = _mm_mul_ps(noise, sigma_vec);
        
        __m128 in_vec = _mm_load_ps(input + i);
        __m128 result = _mm_add_ps(in_vec, noise);
        _mm_store_ps(output + i, result);
    }
    
    // Handle remaining elements
    std::normal_distribution<float> normal_dist(0.0f, sigma);
    for (; i < size; ++i) {
        output[i] = input[i] + normal_dist(gen);
    }
}

void plane_intersection_simd(Vector3f& result,
                           const Vector3f& n1, float d1,
                           const Vector3f& n2, float d2,
                           const Vector3f& n3, float d3) {
    if (!CPUFeatures::hasAVX2()) {
        throw std::runtime_error("AVX2 required for SIMD functions");
    }
    
    alignas(16) float row1[4] = { n1.x(), n1.y(), n1.z(), 0.0f };
    alignas(16) float row2[4] = { n2.x(), n2.y(), n2.z(), 0.0f };
    alignas(16) float row3[4] = { n3.x(), n3.y(), n3.z(), 0.0f };
    
    __m128 r1 = _mm_load_ps(row1);
    __m128 r2 = _mm_load_ps(row2);
    __m128 r3 = _mm_load_ps(row3);
    
    __m128 det = _mm_add_ps(
        _mm_mul_ps(_mm_shuffle_ps(r1, r1, _MM_SHUFFLE(3,0,2,1)),
                   _mm_mul_ps(_mm_shuffle_ps(r2, r2, _MM_SHUFFLE(3,1,0,2)),
                             _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(3,2,1,0)))),
        _mm_mul_ps(_mm_shuffle_ps(r1, r1, _MM_SHUFFLE(3,2,1,0)),
                   _mm_mul_ps(_mm_shuffle_ps(r2, r2, _MM_SHUFFLE(3,1,0,2)),
                             _mm_shuffle_ps(r3, r3, _MM_SHUFFLE(3,0,2,1))))
    );
    
    __m128 inv_det = _mm_rcp_ps(det);
    __m128 d_vec = _mm_set_ps(0.0f, d3, d2, d1);
    __m128 res = _mm_mul_ps(inv_det, d_vec);
    
    alignas(16) float result_data[4];
    _mm_store_ps(result_data, res);
    result = Vector3f(result_data[0], result_data[1], result_data[2]);
}

#include <immintrin.h>   // for AVX2 / SSE intrinsics
#include <stdexcept>
#include <algorithm>
#include <iostream>

// Assuming you have something like:
//   struct Vector2f { float x_, y_; /* ... */ };
//   inline float x() const { return x_; }
//   inline float y() const { return y_; }
//   // plus operators, dot(), squaredNorm(), etc.
// And your Verts3D, Verts2D, Plane, Face, etc. from context.

int count_crossings_simd(const Verts3D& v3ds, const Plane& plane, const Face& poly)
{
    if (!CPUFeatures::hasAVX2()) {
        throw std::runtime_error("AVX2 required for SIMD functions");
    }

    static const float epsilon  = 1e-3f;
    static const float epsilon2 = 1e-10f;

    static Verts2D v2ds;  
    make_2d_projection(v3ds, poly, plane, v2ds);

    const size_t num = poly.size();
    if (num < 2) {
        return 0;
    }

    int crossings = 0;

    // Outer loop: same as scalar
    Vector2f a1 = v2ds[0];
    for (size_t i = 1; i < num; ++i)
    {
        const Vector2f& a2 = v2ds[i];
        const Vector2f da  = a2 - a1;

        // Preload da into an SSE register
        alignas(16) float da_data[4] = { da.x(), da.y(), da.x(), da.y() };
        __m128 da_vec = _mm_load_ps(da_data);

        // Inner loop: replicate scalar iteration pattern
        Vector2f b1 = v2ds[num - 1];
        for (size_t j = 0; j < i - 1; )
        {
            // Adjacency skip, same as scalar
            if (i - j == num - 1) {
                b1 = v2ds[j];
                j++;
                continue;
            }

            // Attempt to process up to 2 edges in one batch
            size_t remaining = std::min<size_t>(2, (i - 1) - j);

            // Gather data for up to 2 edges
            alignas(16) float edge_data[8]; 
            bool skip_edge[2] = { false, false };

            Vector2f local_b1 = b1;
            for (size_t k = 0; k < remaining; ++k)
            {
                // Check adjacency *per edge*
                if (i - (j + k) == num - 1) {
                    skip_edge[k] = true;
                }

                const Vector2f b2 = v2ds[j + k];
                // Pack (b1.x, b1.y, b2.x, b2.y)
                edge_data[k*4 + 0] = local_b1.x();
                edge_data[k*4 + 1] = local_b1.y();
                edge_data[k*4 + 2] = b2.x();
                edge_data[k*4 + 3] = b2.y();

                // In scalar code, b1 = b2 at end of each iteration
                local_b1 = b2;
            }

            // Now do the SIMD pass for these edges
            for (size_t k = 0; k < remaining; ++k)
            {
                // If adjacency skip triggered, replicate scalar’s skip
                if (skip_edge[k]) {
                    b1 = v2ds[j + k];
                    continue;
                }

                // Unpack edge data
                float bx1 = edge_data[k*4 + 0];
                float by1 = edge_data[k*4 + 1];
                float bx2 = edge_data[k*4 + 2];
                float by2 = edge_data[k*4 + 3];

                // db = (bx2 - bx1, by2 - by1)
                float dbx = bx2 - bx1;
                float dby = by2 - by1;
                
                alignas(16) float db_data[4] = { dbx, dby, dbx, dby };
                __m128 db_vec = _mm_load_ps(db_data);

                // determinant = da.x*db.y - da.y*db.x
                __m128 da_x = _mm_shuffle_ps(da_vec, da_vec, _MM_SHUFFLE(0,0,0,0));
                __m128 da_y = _mm_shuffle_ps(da_vec, da_vec, _MM_SHUFFLE(1,1,1,1));
                __m128 db_x = _mm_shuffle_ps(db_vec, db_vec, _MM_SHUFFLE(0,0,0,0));
                __m128 db_y = _mm_shuffle_ps(db_vec, db_vec, _MM_SHUFFLE(1,1,1,1));

                __m128 det_vec = _mm_sub_ps(_mm_mul_ps(da_x, db_y), _mm_mul_ps(da_y, db_x));

                alignas(16) float det_store[4];
                _mm_store_ps(det_store, det_vec);
                float det = det_store[0];  // Each edge is done separately

                // Near-parallel fallback
                if (det * det < epsilon2)
                {
                    // Distance check as in scalar
                    Vector2f p_b1(bx1, by1);
                    Vector2f p_b2(bx2, by2);
                    Vector2f ba = p_b1 - a1;
                    float dotBA_DA = ba.dot(da);
                    float dotDA_DA = da.squaredNorm();
                    // Project ba onto da
                    Vector2f v = ba - da * (dotBA_DA / dotDA_DA);
                    float dist_sq = v.squaredNorm();

                    if (dist_sq < epsilon) {
                        crossings += 1;
                    }
                }
                else
                {
                    // Intersection param check
                    Vector2f p_b1(bx1, by1);
                    Vector2f ba = p_b1 - a1;
                    float t_num = (dby * ba.x()) - (dbx * ba.y());
                    float s_num = (da.x() * ba.y()) - (da.y() * ba.x());
                    float t = t_num / det;
                    float s = s_num / det;

                    if (t > -epsilon && t < 1.f + epsilon &&
                        s > -epsilon && s < 1.f + epsilon)
                    {
                        crossings += 1;
                    }
                }

                // b1 = b2 after each edge (match scalar exactly)
                b1 = v2ds[j + k];
            }

            j += remaining;
        }

        a1 = a2;
    }

    return crossings;
}

// Add similar debug output to the scalar version in util.cpp:
int count_crossings(const Verts3D& v3ds, const Plane& plane, const Face& poly) {
    if (CPUFeatures::hasAVX2()) {
        return simd::count_crossings_simd(v3ds, plane, poly);
    }
    
    static const float epsilon = 1e-3f;
    static const float epsilon2 = 1e-10f;
    static Verts2D v2ds;

    int crossings = 0;
    make_2d_projection(v3ds, poly, plane, v2ds);

    const size_t num = poly.size();
    Vector2f a1 = v2ds[0];
    for (size_t i = 1; i < num; ++i) {
        const Vector2f& a2 = v2ds[i];
        const Vector2f da = a2 - a1;

        Vector2f b1 = v2ds[num - 1];
        for (size_t j = 0; j < i - 1; ++j) {
            if (i - j == num - 1) {
                b1 = v2ds[j];
                continue;
            }

            const Vector2f& b2 = v2ds[j];
            const Vector2f db = b1 - b2;
            const Vector2f ba = b1 - a1;
            const float det = da.x()*db.y() - da.y()*db.x();
            
            std::cout << "Scalar checking lines (" << i-1 << "," << i << ") vs (" << j << "," << j+1 << ")\n";
            std::cout << "  det = " << det << ", det^2 = " << det*det << "\n";
            
            if (det * det < epsilon2) {
                const Vector2f v = ba - da * (ba.dot(da) / da.squaredNorm());
                float dist_sq = v.squaredNorm();
                std::cout << "  parallel case: dist_sq = " << dist_sq << "\n";
                if (dist_sq < epsilon) {
                    std::cout << "  Found crossing (parallel)\n";
                    crossings += 1;
                }
            } else {
                const float t = (db.y()*ba.x() - db.x()*ba.y()) / det;
                const float s = (da.x()*ba.y() - da.y()*ba.x()) / det;
                std::cout << "  intersection params: t = " << t << ", s = " << s << "\n";
                
                if (t > -epsilon && t < 1.0f + epsilon && 
                    s > -epsilon && s < 1.0f + epsilon) {
                    std::cout << "  Found crossing (intersection)\n";
                    crossings += 1;
                }
            }
            b1 = b2;
        }
        a1 = a2;
    }
    return crossings;
}


} // namespace simd