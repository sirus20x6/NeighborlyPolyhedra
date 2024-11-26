//#define USE_CAIRO
#include "util.h"
#include "solver.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <filesystem>
#ifdef USE_CAIRO
#include <cairo.h>
#endif

#define NUM_TOPOLOGIES 59
#define DUAL_PROBLEM 0

#ifdef USE_CAIRO
void render_cutout(const Verts3D& v3ds, const Planes& planes, int width) {
    //Convert to 2D faces
    std::vector<Verts2D> faces(planes.size());
    float cur_x = 0.0f;
    float cur_y = 0.0f;
    float max_x = 0.0f;
    float max_y = 0.0f;
    for (size_t i = 0; i < planes.size(); ++i) {
        //Project points
        make_2d_projection(v3ds, g_polys[i], planes[i], faces[i]);

        //Figure out a bounding box
        Vector2f minCoord(1e9f, 1e9f);
        Vector2f maxCoord(-1e9f, -1e9f);
        for (Vector2f& v : faces[i]) {
            minCoord = minCoord.cwiseMin(v);
            maxCoord = maxCoord.cwiseMax(v);
        }

        //Transform coordinates
        for (Vector2f& v : faces[i]) {
            v -= minCoord;
            v.x() += cur_x;
            v.y() += cur_y;
        }

        //Advance height to next slot
        cur_y += maxCoord.y() - minCoord.y();
        max_x = std::max(max_x, maxCoord.x() - minCoord.x());

        if (i % 3 == 2) {
            cur_x += max_x;
            max_y = std::max(max_y, cur_y);
            cur_y = 0.0f;
            max_x = 0.0f;
        }
    }

    //Compute the scale factor
    const float padding = 4.0f;
    const float scale = float(width - padding*2.0f) / cur_x;

    //Create surface to draw on
    cairo_surface_t* surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, int(max_y * scale + padding * 2.0f));
    cairo_t* cr = cairo_create(surface);

    cairo_set_line_width(cr, 2.0);
    cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);

    for (const Verts2D& face : faces) {
      const Vector2f& start_pt = face[face.size() - 1];
      cairo_move_to(cr, double(start_pt.x() * scale + padding), double(start_pt.y() * scale + padding));
      for (const Vector2f& p : face) {
        cairo_line_to(cr, double(p.x() * scale + padding), double(p.y() * scale + padding));
        cairo_stroke(cr);
        cairo_move_to(cr, double(p.x() * scale + padding), double(p.y() * scale + padding));
      }
    }

    //Save the image and free all memory
    cairo_surface_write_to_png(surface, "face1.png");
    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}
#endif

void validate_files(bool change_files=true) {
    for (int i = 0; i < NUM_TOPOLOGIES; ++i) {
#if DUAL_PROBLEM != 0
        const std::filesystem::path dir("../Szilassi/Dual/topology_" + std::to_string(i));
#else
        const std::filesystem::path dir("../Szilassi/topology_" + std::to_string(i));
#endif
        for (const auto& dir_entry : std::filesystem::directory_iterator(dir)) {
            //Get intersections and crossings reported from file
            const auto& path = dir_entry.path();
            if (path.extension() != ".obj") { continue; }
            std::vector<std::string> name_split = split(path.filename().string(), '_');
            const int f_crossings = std::stoi(name_split[1].substr(1));
            const int f_intersections = std::stoi(name_split[2].substr(1));

            //Check for nan or inf in file
            Verts3D obj_verts;
            g_topology = i;
            import_obj(path.string().c_str(), obj_verts, g_polys);
            if (!is_finite(obj_verts)) {
                std::cout << path << std::endl;
                std::cout << "  NaNs found!" << std::endl << std::endl;
                if (change_files) { std::filesystem::remove(path); }
                continue;
            }

            //Get actual intersections and crossings
            Edges dual_edges;
            Planes obj_planes;
            make_edges(g_polys, g_edges);
            dual_graph(g_polys, g_tris, dual_edges);
            v3ds_to_planes(obj_verts, g_polys, obj_planes);
            const int crossings = count_crossings(obj_verts, obj_planes);
            const int intersections = count_intersections(obj_verts, obj_planes);

            //Check degenerate scores for study samples
            if (name_split[0] == "study") {
              const float dp = dist_penalty(obj_verts);
              const float lp = length_penalty(obj_verts);
              const float ap = angle_penalty(obj_verts);
              const float pp = plane_penalty(obj_planes);
              if (dp >= 0.9999f || ap >= 0.9999f || lp >= 0.9999f || pp >= 0.9999f) {
                std::cout << path << std::endl;
                std::cout << "  DEGENERATE! dp(" << dp << ") dp(" << ap << ") lp(" << lp << ") pp(" << pp << ")" << std::endl << std::endl;
                if (change_files) { std::filesystem::remove(path); }
                continue;
              }
            }

            //Compare
            if (crossings != f_crossings || intersections != f_intersections) {
                const std::string new_name = name_split[0] + "_c" + std::to_string(crossings) + "_i" + std::to_string(intersections) + "_" + name_split[3];
                const std::filesystem::path new_path = std::filesystem::path(path).replace_filename(new_name);
                std::cout << path << std::endl;
                std::cout << new_path << std::endl << std::endl;
                if (change_files) { std::filesystem::rename(path, new_path); }
                continue;
            }
        }
    }
}

void main_solver() {
    int iter = 0;
    while (true) {
        //Load the dual adjacency list
        g_topology = iter % NUM_TOPOLOGIES;
        open_topology("topologies.txt", g_tris, g_topology);
        std::cout << "Topology[" << g_topology << "]" << std::endl;

        //Create directory for results
#if DUAL_PROBLEM != 0
        const std::string topology_folder = "Dual/topology_" + std::to_string(g_topology);
#else
        const std::string topology_folder = "topology_" + std::to_string(g_topology);
#endif
        if (!std::filesystem::exists(topology_folder)) {
            std::filesystem::create_directory(topology_folder);
        }

        //Find the dual graph to get the polygon and edge linkage
        dual_graph(g_tris, g_polys, g_edges);
        fix_face_ordering(g_polys, g_edges);
#if DUAL_PROBLEM != 0
        std::swap(g_tris, g_polys);
        make_edges(g_polys, g_edges);
        fix_face_ordering(g_polys, g_edges);
#endif

        //Run the optimizer
        VectorXf result;
        float score = my_optimizer(objective_sum, result, 16000, 0.5f, 0.998f, 32, false, DUAL_PROBLEM);

        //Get the actual values of crossings and intersection independent of score
        Planes planes;
        Verts3D v3ds;
#if DUAL_PROBLEM != 0
        y_to_v3ds(result, v3ds);
        v3ds_to_planes(v3ds, g_polys, planes);
#else
        x_to_planes(result, planes);
        planes_to_v3ds(g_tris, planes, v3ds);
#endif
        const int crossings = count_crossings(v3ds, planes);
        const int intersections = count_intersections(v3ds, planes);

        //Check we should save it
        std::cout << "Score    : " << score << std::endl;
#if DUAL_PROBLEM != 0
        const bool can_save = (intersections <= 8);
#else
        const bool can_save = (crossings == 0 || (crossings + intersections <= 10));
#endif
        const std::string save_str = topology_folder + "/shape";
        save_sample(save_str.c_str(), planes, v3ds, iter, can_save);
        iter += 1;
    }
}

void quality_solver() {
    int iter = 0;
    while (true) {
        iter += 1;
        g_topology = iter % NUM_TOPOLOGIES;
#if DUAL_PROBLEM != 0
        const std::string topology_folder = "Dual/topology_" + std::to_string(g_topology);
#else
        const std::string topology_folder = "topology_" + std::to_string(g_topology);
#endif
        const std::filesystem::path dir("../Szilassi/" + topology_folder);
        std::vector<std::filesystem::path> paths;
        for (const auto& dir_entry : std::filesystem::directory_iterator(dir)) {
            //Get intersections and crossings reported from file
            const auto& path = dir_entry.path();
            if (path.extension() != ".obj") { continue; }
            std::vector<std::string> name_split = split(path.filename().string(), '_');
            if (name_split[0] != "shape") { continue; }
            const int f_crossings = std::stoi(name_split[1].substr(1));
            const int f_intersections = std::stoi(name_split[2].substr(1));
            if ((f_crossings == 0 && f_intersections <= 12) ||
                (f_crossings <= 1 && f_intersections <= 8) ||
                (f_crossings <= 2 && f_intersections <= 6) ||
                (f_crossings <= 4 && f_intersections <= 4)) {
                paths.push_back(path);
            }
        }

        if (paths.size() == 0) { continue; }
        const std::filesystem::path& fpath = paths[std::uniform_int_distribution<int>(0, (int)paths.size() - 1)(eng)];

        Verts3D obj_verts;
        Edges dual_edges;
        Planes obj_planes;
        VectorXf obj_x;
        std::vector<std::string> name_split = split(fpath.stem().string(), '_');
        const int f_iter = std::atoi(name_split[name_split.size() - 1].c_str());
        import_obj(fpath.string().c_str(), obj_verts, g_polys);
        make_edges(g_polys, g_edges);
        dual_graph(g_polys, g_tris, dual_edges);
        v3ds_to_planes(obj_verts, g_polys, obj_planes);
#if DUAL_PROBLEM != 0
        v3ds_to_y(obj_verts, obj_x);
#else
        planes_to_x(obj_planes, obj_x);
#endif

        //Print characteristics
        std::cout << "===================" << std::endl;
        std::cout << "Loaded:        " << fpath << std::endl;
        save_sample("study", obj_planes, obj_verts, f_iter, false);
        std::cout << "===================" << std::endl;

        //Run optimizer
        study_sample(objective_dual_q, obj_x, 360, 10000, 1e-2f, 0.9f, true);
#if DUAL_PROBLEM != 0
        y_to_v3ds(obj_x, obj_verts);
        v3ds_to_planes(obj_verts, g_polys, obj_planes);
#else
        x_to_planes(obj_x, obj_planes);
        planes_to_v3ds(g_tris, obj_planes, obj_verts);
#endif

        if (q_penalty(obj_verts) >= 0.9999f) {
            std::cout << "DEGENERATE" << std::endl;
        } else {
            const std::string out_path = fpath.parent_path().string() + "/study";
            save_sample(out_path.c_str(), obj_planes, obj_verts, f_iter, true);
            //render_cutout(obj_verts, obj_planes, 3840);
        }
    }
}

void explore_shape(const char* load_fname) {
    //Import an example obj file
    g_topology = 6;
    Verts3D obj_verts;
    Edges dual_edges;
    Planes obj_planes;
    VectorXf obj_x;
    std::vector<std::string> name_split = split(std::filesystem::path(load_fname).stem().string(), '_');
    const int f_iter = std::atoi(name_split[name_split.size() - 1].c_str());
    import_obj(load_fname, obj_verts, g_polys);
    make_edges(g_polys, g_edges);
    dual_graph(g_polys, g_tris, dual_edges);
    v3ds_to_planes(obj_verts, g_polys, obj_planes);
#if DUAL_PROBLEM != 0
    v3ds_to_y(obj_verts, obj_x);
#else
    planes_to_x(obj_planes, obj_x);
#endif

    //Print characteristics
    std::cout << "===================" << std::endl;
    std::cout << "Loaded:        " << load_fname << std::endl;
    save_sample("study", obj_planes, obj_verts, f_iter, false);
    std::cout << "===================" << std::endl;

    //Run optimizer
    study_sample(objective_sum_q, obj_x, 360, 10000, 1e-2f, 0.9f, true);
#if DUAL_PROBLEM != 0
    y_to_v3ds(obj_x, obj_verts);
    v3ds_to_planes(obj_verts, g_polys, obj_planes);
#else
    x_to_planes(obj_x, obj_planes);
    planes_to_v3ds(g_tris, obj_planes, obj_verts);
#endif
    save_sample("study", obj_planes, obj_verts, f_iter, true);
#ifdef USE_CAIRO
    render_cutout(obj_verts, obj_planes, 3840);
#endif
}

int main(int argc, char* argv[]) {
    //validate_files(true);
    //return 0;

    int seed = 123;
    std::cout << "Seed: ";
    std::cin >> seed;
    std::cout << std::endl;
    set_rand_seed(seed);

    //Run the optimizer (choose one)
    main_solver();
    //quality_solver();
    //explore_shape("../Szilassi/topology_42/shape_c0_i4_optimalsymmetric.obj");

    return 0;
}
