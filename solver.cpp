#include "solver.h"
#include <algorithm>
#include <iostream>

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

void set_rand_seed(int seed) {
  eng.seed(seed);
}

void add_random_noise(const VectorXf& x, float sigma, VectorXf& result) {
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

float my_optimizer(float (*objective_function)(const VectorXf&), VectorXf& result, int max_iters,
                   float sigma, float beta, int clusters, bool use_symmetry, bool dual_problem) {
    static const int extra_tries = 10;
    std::vector<std::pair<float, VectorXf>> xv(clusters * extra_tries);
    VectorXf new_pt;

    initialize_random_noise(xv, 1.0f, dual_problem);
    for (auto& x : xv) {
        if (use_symmetry) { SYMMETRY(x.second); }
        x.first = objective_function(x.second);
    }
    std::sort(xv.begin(), xv.end(), v_pred);
    xv.resize(clusters);

    int iter = 0;
    float best_cost = 99999;
    const size_t x_size = xv[0].second.size();
    new_pt.resize(x_size);
    while (true) {
        const size_t min_ix = std::min_element(xv.begin(), xv.end(), v_pred) - xv.begin();
        const float min_cost = xv[min_ix].first;
        iter += 1;
        if (min_cost < best_cost) {
            std::cout << min_cost << "    " << iter << std::endl;
            best_cost = min_cost;
            result = xv[min_ix].second;
        }

#if 0
        if (best_cost > 500 && iter > max_iters) { break; }
        if (best_cost > 300 && iter > max_iters * 2) { break; }
        if (best_cost > 200 && iter > max_iters * 3) { break; }
        if (best_cost > 100 && iter > max_iters * 4) { break; }
#elif 0
        if (best_cost > 22 && iter > max_iters) { break; }
        if (best_cost > 18 && iter > max_iters * 2) { break; }
        if (best_cost > 15 && iter > max_iters * 3) { break; }
        if (best_cost > 12 && iter > max_iters * 4) { break; }
#else
        if (best_cost > 11 && iter > max_iters) { break; }
        if (best_cost > 9 && iter > max_iters * 2) { break; }
        if (best_cost > 7 && iter > max_iters * 3) { break; }
        if (best_cost > 5 && iter > max_iters * 4) { break; }
#endif
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
            const float new_cost = objective_function(new_pt);
            const float cost_mult = (i == min_ix ? 1.0f : 1.25f);
            if (new_cost <= best_cost * cost_mult || new_cost < xv[i].first) {
                xv[i].first = new_cost;
                xv[i].second = new_pt;
            }
        }
    }
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
