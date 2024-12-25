#include "solver.h"
#include "cpu_features.h"
#include "simd_util.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iomanip>

#define SYMMETRY make_symmetric_o4
//#define SYMMETRY make_symmetric_o2
//#define SYMMETRY make_symmetric_d2

RNG eng;

struct Sample {
    VectorXf x;
    float score;
    int time;

    bool operator<(const Sample& rhs) const {
        return this->score < rhs.score;
    }
};

void PerformanceStats::print() const {
    if (iterations == 0) return;
    
    double avg_iter_time = total_time / iterations;
    double avg_obj_time = objective_time / objective_calls;
    
    // Calculate iteration time statistics
    std::vector<double> iter_times = iteration_times;
    std::sort(iter_times.begin(), iter_times.end());
    double median_iter_time = iter_times[iter_times.size() / 2];
    double p95_iter_time = iter_times[size_t(iter_times.size() * 0.95)];
    
    std::cout << "\n=== Performance Statistics ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total time: " << total_time << " sec\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Iterations/sec: " << iterations / total_time << "\n";
    std::cout << "Avg iteration time: " << avg_iter_time * 1000 << " ms\n";
    std::cout << "Median iteration time: " << median_iter_time * 1000 << " ms\n";
    std::cout << "P95 iteration time: " << p95_iter_time * 1000 << " ms\n";
    std::cout << "Objective calls: " << objective_calls << "\n";
    std::cout << "Avg objective time: " << avg_obj_time * 1000 << " ms\n";
    std::cout << "Objective time %: " << (objective_time / total_time) * 100 << "%\n";
    std::cout << "===========================\n\n";
}

void set_rand_seed(int seed) {
  eng.seed(seed);
}

void add_random_noise(const VectorXf& x, float sigma, VectorXf& result) {
    result.resize(x.size());
    
    if (CPUFeatures::hasAVX2()) {
        simd::add_random_noise_simd(x.data(), sigma, result.data(), x.size());
        return;
    }
    
    // Original non-SIMD implementation
    std::normal_distribution<float> rand_normal(0.0f, sigma);
    for (int i = 0; i < x.size(); ++i) {
        result[i] = x[i] + rand_normal(eng);
    }
}

void set_random_mags(VectorXf& x, float sigma) {
  std::normal_distribution<float> rand_normal(0.0f, sigma);
  for (int i = 0; i < x.size(); i += 3) {
    Eigen::Map<Vector3f> sub_x(x.data() + i);
    sub_x.normalize();
    sub_x *= rand_normal(eng);
  }
}

void initialize_random_noise(VectorXf& x, float sigma, size_t size) {
    std::normal_distribution<float> rand_normal(0.0f, sigma);
    x.resize(size);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = rand_normal(eng);
    }
}

void initialize_random_noise(std::vector<std::pair<float, VectorXf>>& xvec, float sigma, bool dual_problem) {
    size_t size = (dual_problem ? g_tris.size() : g_polys.size()) * 3;
    for (auto& x : xvec) {
        initialize_random_noise(x.second, sigma, size);
    }
}

void initialize_random_noise(std::vector<std::pair<float, VectorXf>>& xvec, const VectorXf& guess, float sigma) {
    for (auto& x : xvec) {
        initialize_random_noise(x.second, sigma, guess.size());
        x.second += guess;
    }
}

void initialize_random_mag(std::vector<VectorXf>& yvec, const VectorXf& n, float sigma) {
    std::normal_distribution<float> rand_normal(0.0f, sigma);
    for (VectorXf& y : yvec) {
        y.resize(g_polys.size());
        for (int i = 0; i < y.size(); ++i) {
            y[i] = rand_normal(eng);
        }
    }
}

VectorXf shuffle_x(const VectorXf& x) {
    VectorXf result(x);
    const size_t num = x.size() / 3;
    for (size_t i = 0; i < num; ++i) {
        const size_t ix = std::uniform_int_distribution<size_t>(i, num - 1)(eng);
        std::swap(result[ix*3 + 0], result[i*3 + 0]);
        std::swap(result[ix*3 + 1], result[i*3 + 1]);
        std::swap(result[ix*3 + 2], result[i*3 + 2]);
    }
    return result;
}

VectorXf random_plane_shift(const VectorXf& x) {
    std::uniform_real_distribution<float> rand_uniform(0.8f, 1.2f);
    VectorXf result(x);
    const size_t num = x.size() / 3;
    const size_t ix = std::uniform_int_distribution<size_t>(0, num - 1)(eng);
    const float scale = rand_uniform(eng);
    result[ix * 3 + 0] *= scale;
    result[ix * 3 + 1] *= scale;
    result[ix * 3 + 2] *= scale;
    return result;
}

void z_symmetric(VectorXf& x, size_t i, size_t j) {
  x[j*3 + 0] = -x[i*3 + 0];
  x[j*3 + 1] = -x[i*3 + 1];
  x[j*3 + 2] =  x[i*3 + 2];
}

void z_rotate_inv(VectorXf& x, size_t i, size_t j) {
  x[j * 3 + 0] =  x[i * 3 + 1];
  x[j * 3 + 1] = -x[i * 3 + 0];
  x[j * 3 + 2] = -x[i * 3 + 2];
}

void make_symmetric_d2(VectorXf& x) {
  z_symmetric(x, 0, 1);
  z_symmetric(x, 2, 3);
  z_symmetric(x, 4, 5);
  z_symmetric(x, 6, 7);
  z_symmetric(x, 8, 9);
  z_symmetric(x, 10, 11);
}

void make_symmetric_o2(VectorXf& x) {
  z_symmetric(x, 0, 2);
  z_symmetric(x, 5, 7);
  z_symmetric(x, 8, 10);
  z_symmetric(x, 1, 3);
  z_symmetric(x, 4, 6);
  z_symmetric(x, 9, 11);
}

void make_symmetric_o4(VectorXf& x) {
  z_symmetric(x, 0, 2);
  z_symmetric(x, 5, 7);
  z_symmetric(x, 8, 10);
  z_rotate_inv(x, 5, 4);
  z_rotate_inv(x, 7, 6);
  z_rotate_inv(x, 0, 3);
  z_rotate_inv(x, 2, 1);
  z_rotate_inv(x, 8, 11);
  z_rotate_inv(x, 10, 9);
}

bool v_pred(const std::pair<float, VectorXf>& left, const std::pair<float, VectorXf>& right) {
    return left.first < right.first;
}

float my_optimizer(float (*objective_function)(const VectorXf&),
                   VectorXf& result,
                   int max_iters,
                   float sigma,
                   float beta,
                   int clusters,
                   bool use_symmetry,
                   bool dual_problem)
{
    static const int extra_tries = 10;
    std::vector<std::pair<float, VectorXf>> xv(clusters * extra_tries);
    VectorXf new_pt;
    PerformanceStats stats;
    auto total_start = std::chrono::high_resolution_clock::now();

    // Wrap objective function to track timing
    TimedObjective timed_obj(objective_function, stats);

    initialize_random_noise(xv, 1.0f, dual_problem);
    for (auto& x : xv) {
        if (use_symmetry) { SYMMETRY(x.second); }
        x.first = timed_obj(x.second);
    }
    std::sort(xv.begin(), xv.end(), v_pred);
    xv.resize(clusters);

    int iter = 0;
    float best_cost = 99999;
    const size_t x_size = xv[0].second.size();
    new_pt.resize(x_size);

    //-----------------------------------------------------------------//
    // Print a table header just once at the start.
    // Adjust widths and alignment as needed.
    //-----------------------------------------------------------------//
    std::cout << std::format("{:^10} | {:^6} | {:^12} | {:^14}\n",
                             "Cost", "Iter", "Time (s)", "Iter/s");
    std::cout << std::string(10 + 1 + 6 + 1 + 12 + 1 + 14, '-') << "\n";
    //-----------------------------------------------------------------//

    while (true)
    {
        auto iter_start = std::chrono::high_resolution_clock::now();

        const size_t min_ix = std::min_element(xv.begin(), xv.end(), v_pred) - xv.begin();
        const float min_cost = xv[min_ix].first;
        iter += 1;
        stats.iterations = iter;

        if (min_cost < best_cost)
        {
            auto current = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current - total_start;
            best_cost = min_cost;
            result = xv[min_ix].second;

            //-----------------------------------------------------------------//
            // Instead of:
            // std::cout << min_cost << "    " << iter << "    " << elapsed.count()
            //           << "s    " << iter / elapsed.count() << " it/s\n";
            //
            // We do:
            //-----------------------------------------------------------------//
            std::cout << std::format("{:>10.4f} | {:>6} | {:>12.5f} | {:>14.5f}\n",
                                     min_cost,
                                     iter,
                                     elapsed.count(),
                                     static_cast<double>(iter) / elapsed.count());
        }

        // Early exit conditions
        if (best_cost > 11 && iter > max_iters) { break; }
        if (best_cost > 9 && iter > max_iters * 2) { break; }
        if (best_cost > 7 && iter > max_iters * 3) { break; }
        if (best_cost > 5 && iter > max_iters * 4) { break; }
        if (iter > max_iters * 10) { break; }
        if (best_cost == 0.0f) { break; }

        for (size_t i = 0; i < clusters; ++i) {
            size_t ix = i;
            if (xv[i].first > best_cost * 1.25f) {
                ix = std::uniform_int_distribution<size_t>(0, clusters-1)(eng);
            }
            add_random_noise(xv[ix].second, sigma, new_pt);
            if (use_symmetry) { SYMMETRY(new_pt); }
            new_pt *= beta;
            const float new_cost = timed_obj(new_pt);
            const float cost_mult = (i == min_ix ? 1.0f : 1.25f);
            if (new_cost <= best_cost * cost_mult || new_cost < xv[i].first) {
                xv[i].first = new_cost;
                xv[i].second = new_pt;
            }
        }

        auto iter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> iter_time = iter_end - iter_start;
        stats.iteration_times.push_back(iter_time.count());
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = total_end - total_start;
    stats.total_time = total_time.count();

    // Optionally print final stats summary
    stats.print();

    return best_cost;
}

void study_sample(float (*objective_function)(const VectorXf&), VectorXf& result, int max_iters, int clusters, float sigma, float beta, bool use_symmetry) {
    const size_t x_size = result.size();
    if (use_symmetry) { SYMMETRY(result); }
    float min_score = objective_function(result);
    std::vector<VectorXf> xv(clusters, result);
    std::vector<float> scores(clusters, min_score);
    VectorXf new_x(x_size);

    for(int iter = 0; iter < max_iters; ++iter) {
        int num_updated = 0;
        for (int i = 0; i < clusters; ++i) {
            add_random_noise(xv[i], sigma, new_x);
            if (use_symmetry) { SYMMETRY(new_x); }
            new_x.normalize();
            const float new_cost = objective_function(new_x);
            if (new_cost < scores[i]) {
                if (new_cost < min_score) {
                    std::cout << "==== New Best! ==== (" << new_cost << ")" << std::endl;
                    result = new_x;
                    if (int(new_cost) < int(min_score)) {
                        std::fill(xv.begin(), xv.end(), new_x);
                    }
                    min_score = new_cost;
                }
                xv[i] = new_x;
                scores[i] = new_cost;
                num_updated += 1;
            }
        }
        std::cout << "Updated: " << num_updated << "/" << clusters << "      " << sigma << std::endl;
        if (num_updated < clusters / 100) {
            sigma *= beta;
        } else if (num_updated > clusters / 10) {
            sigma *= 1.01f;
        }
        if (sigma < 1e-7f) { break; }
    }
}
