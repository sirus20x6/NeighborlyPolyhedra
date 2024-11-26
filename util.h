#pragma once
#include "plane.h"
#include <Eigen/Dense>
#include <unordered_set>
#include <vector>
#include <map>

using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector3d;
using Eigen::VectorXf;
using Eigen::Matrix3f;
using Eigen::Matrix3d;

using Verts2D = std::vector<Vector2f>;
using Verts3D = std::vector<Vector3f>;
using Planes = std::vector<Plane>;
using Face = std::vector<int>;
using Faces = std::vector<Face>;
using Edge = std::pair<int, int>;
using Edges = std::vector<Edge>;

using FaceMap = std::map<int, std::vector<int>>;
using EdgeMap = std::map<Edge, std::vector<int>>;

std::vector<std::string> split(const std::string& s, char delim);
bool is_finite(const Verts3D& verts);
void print_faces(const Faces& faces);
void open_face_file(const char* fname, Faces& tris);
void open_topology(const char* fname, Faces& tris, int ix);
bool verify_topology(const Faces& tris);
void dual_graph(const Faces& faces, Faces& dual_faces, Edges& dual_edges);
void save_sample(const char* name, const Planes& planes, const Verts3D& verts, int iter, bool can_save);
void save_dot_graph(const char* fname, const Edges& edges);
void make_edges(const Faces& faces, Edges& edges);
void fix_face_ordering(Faces& polys, const Edges& edges);
bool test_face_ordering(const Faces& polys);
void import_obj(const char* fname, Verts3D& verts, Faces& polys);
void export_obj(const char* fname, const Verts3D& verts, const Faces& polys);
void line_line_intersection(const Vector3f& a1, const Vector3f& a2, const Vector3f& b1, const Vector3f& b2, Vector3f& pa, Vector3f& pb);

Plane get_plane(const Verts3D& pts, const Face& poly);
void make_2d_projection(const Verts3D v3ds, const Face& poly, const Plane& plane, Verts2D& v2ds);
void make_2d_projection(const Verts3D v3ds, const Face& poly, const Plane& plane, Verts2D& v2ds, Matrix3f& basis, Vector3f& p);
Vector3f plane_intersection(const Plane& p1, const Plane& p2, const Plane& p3);
void spread_planes(VectorXf& x, float spread, int iters);

void y_to_v3ds(const VectorXf& y, Verts3D& verts);
void v3ds_to_y(const Verts3D& verts, VectorXf& y);
void x_to_planes(const VectorXf& x, Planes& planes);
void planes_to_x(const Planes& planes, VectorXf& x);
void v3ds_to_planes(const Verts3D& pts, const Faces& polys, Planes& planes);
void planes_to_v3ds(const Faces& dual_tris, const Planes& planes, Verts3D& verts);
void x_to_v3ds(const VectorXf& x, const Faces& tris, Verts3D& verts);
void v3ds_to_x(const Verts3D& pts, const Faces& polys, VectorXf& x);
void y_to_x(const VectorXf& n, const VectorXf& y, VectorXf& x);

bool point_in_polygon(const Vector2f& p, const Verts2D& pts, int& onEdge);
int count_crossings(const Verts3D& v3ds, const Plane& plane, const Face& poly);
int count_crossings(const Verts3D& v3ds, const Planes& planes);
int count_intersections(const Verts3D& v3ds, const Planes& planes, const Plane& plane, const Face& poly, const Edges& other_edges);
int count_intersections(const Verts3D& v3ds, const Planes& planes);
float angle_penalty(const Verts3D& verts);
float dist_penalty(const Verts3D& verts);
float length_penalty(const Verts3D& verts);
float plane_penalty(const Planes& planes);
float q_penalty(const Verts3D& verts);

float objective_cross_int_qlim(const VectorXf& x);
float objective_cross_int_q(const VectorXf& x);
float objective_cross_zint(const VectorXf& x);
float objective_cross_int(const VectorXf& x);
float objective_cross(const VectorXf& x);
float objective_int_cross_q(const VectorXf& x);
float objective_int_cross(const VectorXf& x);
float objective_int_zcross_q(const VectorXf& x);
float objective_int_zcross(const VectorXf& x);
float objective_int_qlim(const VectorXf& x);
float objective_int_q(const VectorXf& x);
float objective_int(const VectorXf& x);
float objective_sum_qlim(const VectorXf& x);
float objective_sum_q(const VectorXf& x);
float objective_sum(const VectorXf& x);
float objective_wsum(const VectorXf& x);
float objective_wsum2(const VectorXf& x);
float objective_wsum_q(const VectorXf& x);

float objective_dual(const VectorXf& x);
float objective_dual_q(const VectorXf& x);

extern Faces g_polys;
extern Faces g_tris;
extern Edges g_edges;
extern int g_topology;
