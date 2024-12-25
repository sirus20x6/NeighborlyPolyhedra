#pragma once
#include "util.h"
#include <random>
#include <chrono>
#include <vector>

//using RNG = std::mt19937;
using RNG = std::minstd_rand;
extern RNG eng;

struct PerformanceStats {
    double total_time = 0.0;
    double objective_time = 0.0;
    int iterations = 0;
    int objective_calls = 0;
    std::vector<double> iteration_times;
    
    void print() const;
};

// Wrapper for objective function to track timing
template<typename F>
class TimedObjective {
    F func;
    PerformanceStats& stats;
    
public:
    TimedObjective(F f, PerformanceStats& s) : func(f), stats(s) {}
    
    float operator()(const VectorXf& x) {
        auto start = std::chrono::high_resolution_clock::now();
        float result = func(x);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        stats.objective_time += diff.count();
        stats.objective_calls++;
        
        return result;
    }
};

void set_rand_seed(int seed);
void add_random_noise(const VectorXf& x, float sigma, VectorXf& result);
void set_random_mags(VectorXf& x, float sigma);
void initialize_random_noise(VectorXf& x, float sigma, size_t size);
void initialize_random_noise(std::vector<std::pair<float, VectorXf>>& xvec, float sigma, bool dual_problem);
void initialize_random_noise(std::vector<std::pair<float, VectorXf>>& xvec, const VectorXf& guess, float sigma);
VectorXf shuffle_x(const VectorXf& x);
void make_symmetric_o4(VectorXf& x);

float my_optimizer(float (*objective_function)(const VectorXf&), VectorXf& result, int max_iters,
                   float sigma, float beta, int clusters, bool use_symmetry, bool dual_problem);
void study_sample(float (*objective_function)(const VectorXf&), VectorXf& result, int max_iters,
                  int clusters, float sigma, float beta, bool use_symmetry);
