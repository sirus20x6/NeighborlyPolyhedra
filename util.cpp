#include "util.h"
#include <set>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

Faces g_polys;
Faces g_tris;
Edges g_edges;
int g_topology = 0;

inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
}

inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
}

std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (getline(ss, item, delim)) {
    result.push_back(item);
  }
  return result;
}

bool is_finite(const Verts3D& verts) {
    for (const Vector3f& v : verts) {
        if (!v.allFinite()) {
            return false;
        }
    }
    return true;
}

void print_faces(const Faces& faces) {
  for (const Face& face : faces) {
    std::cout << "[";
    for (size_t ix = 0; ix < face.size(); ++ix) {
      if (ix > 0) { std::cout << " "; }
      std::cout << face[ix];
    }
    std::cout << "] ";
  }
  std::cout << std::endl;
}

void open_face_file(const char* fname, Faces& tris) {
    std::ifstream fin(fname);
    tris.clear();
    std::string line;
    while (std::getline(fin, line)) {
        Face face;
        ltrim(line);
        rtrim(line);
        std::vector<std::string> strs = split(line, ' ');
        for (const std::string& str : strs) {
            if (str.size() == 0) { continue; }
            face.push_back(std::stoi(str) - 1);
        }
        tris.push_back(face);
    }
    std::cout << "Loaded " << tris.size() << " triangles." << std::endl;
}

inline int alphanumeric_to_int(char c) {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'z') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'Z') { return c - 'a' + 10; }
    return -1;
}

void open_topology(const char* fname, Faces& tris, int ix) {
    std::ifstream fin(fname);
    tris.clear();
    std::string line;
    for (int i = 0; i <= ix; ++i) {
        std::getline(fin, line);
    }
    std::vector<std::string> strs = split(line, ' ');
    for (size_t i = 1; i < strs.size(); ++i) {
        const std::string& face_txt = strs[i];
        const int f1 = alphanumeric_to_int(face_txt[0]);
        const int f2 = alphanumeric_to_int(face_txt[1]);
        const int f3 = alphanumeric_to_int(face_txt[2]);
        tris.push_back({f1, f2, f3});
    }
}

bool verify_topology(const Faces& faces) {
    std::map<int, int> vertex_count;
    std::map<Edge, int> edge_count;
    for (const Face& face : faces) {
        for (size_t i = 0; i < face.size(); ++i) {
            const int v1 = face[i];
            const int v2 = face[(i + 1) % face.size()];
            vertex_count[v1] += 1;
            const int min_v = std::min(v1, v2);
            const int max_v = std::max(v1, v2);
            edge_count[Edge(min_v, max_v)] += 1;
        }
    }
    if (edge_count.size() != faces[0].size() * faces.size() / 2) {
        return false;
    }
    for (auto const& x : edge_count) {
        if (x.second != 2) {
            return false;
        }
    }
    const size_t expected_vcount = (faces[0].size() == 3 ? vertex_count.size() - 1 : 3);
    for (auto const& x : vertex_count) {
        if (x.second != expected_vcount) {
            return false;
        }
    }
    return true;
}

void save_sample(const char* name, const Planes& planes, const Verts3D& verts, int iter, bool can_save) {
    const int num_crossings = count_crossings(verts, planes);
    const int num_intersections = count_intersections(verts, planes);
    std::cout << "Topology : " << g_topology << std::endl;
    std::cout << "Iter     : " << iter << std::endl;
    std::cout << "Crossing : " << num_crossings << std::endl;
    std::cout << "Intersect: " << num_intersections << std::endl;
    std::cout << "A-Factor : " << angle_penalty(verts) << std::endl;
    std::cout << "D-Factor : " << dist_penalty(verts) << std::endl;
    std::cout << "L-Factor : " << length_penalty(verts) << std::endl;
    std::cout << "P-Factor : " << plane_penalty(planes) << std::endl;
    if (can_save) {
        std::cout << "============ Exporting... ============" << std::endl;
        std::stringstream ss;
        ss << name << "_c" << int(num_crossings) << +"_i" << num_intersections << "_" << iter << ".obj";
        export_obj(ss.str().c_str(), verts, g_polys);
    }
}

void save_dot_graph(const char* fname, const Edges& edges) {
    std::ofstream fout(fname);
    fout << "graph nodes {\n";
    for (const Edge& edge : edges) {
        fout << "  N" << edge.first << " -- N" << edge.second << ";\n";
    }
    fout << "}\n";
}

void dual_graph(const Faces& faces, Faces& dual_faces, Edges& dual_edges) {
    FaceMap verts;
    EdgeMap edges;
    for (size_t fIx = 0; fIx < faces.size(); ++fIx) {
        const Face& face = faces[fIx];
        for (size_t vIx = 0; vIx < face.size(); ++vIx) {
            const int v1 = face[vIx];
            const int v2 = face[(vIx + 1) % face.size()];
            verts[v1].push_back((int)fIx);
            const Edge edge(std::min(v1, v2), std::max(v1, v2));
            edges[edge].push_back((int)fIx);
        }
    }
    dual_faces.clear();
    for (FaceMap::const_iterator it = verts.begin(); it != verts.end(); ++it) {
      dual_faces.push_back(it->second);
    }
    dual_edges.clear();
    for (EdgeMap::const_iterator it = edges.begin(); it != edges.end(); ++it) {
      dual_edges.push_back(Edge(it->second[0], it->second[1]));
    }
}

void make_edges(const Faces& faces, Edges& edges) {
    edges.clear();
    for (size_t fIx = 0; fIx < faces.size(); ++fIx) {
        const Face& face = faces[fIx];
        for (size_t vIx = 0; vIx < face.size(); ++vIx) {
              const int v1 = face[vIx];
              const int v2 = face[(vIx + 1) % face.size()];
              const Edge edge(std::min(v1, v2), std::max(v1, v2));
              if (std::find(edges.begin(), edges.end(), edge) == edges.end()) {
                  edges.push_back(edge);
              }
        }
    }
}

bool test_face_ordering(const Faces& polys) {
    std::set<Edge> edges;
    for (const Face& poly : polys) {
        int prevIx = poly[poly.size() - 1];
        for (int ix : poly) {
            const Edge edge(ix, prevIx);
            if (edges.count(edge) > 0) {
                std::cout << edge.first << "," << edge.second << std::endl;
                return false;
            }
            edges.insert(edge);
            prevIx = ix;
        }
    }
    return true;
}

void fix_face_ordering(Faces& polys, const Edges& edges) {
    Faces fixed_polys;
    for (const Face& poly : polys) {
        Face fixed_poly;
        fixed_poly.push_back(poly[0]);
        std::unordered_set<int> poly_set(poly.begin(), poly.end());
        while (fixed_poly.size() < poly.size()) {
            for (const Edge& e : edges) {
                const int e1 = e.first;
                const int e2 = e.second;
                const int lasVert = fixed_poly[fixed_poly.size() - 1];
                if (e1 == lasVert && poly_set.find(e2) != poly_set.end() &&
                    std::find(fixed_poly.begin(), fixed_poly.end(), e2) == fixed_poly.end()) {
                    fixed_poly.push_back(e2);
                    break;
                }
                if (e2 == lasVert && poly_set.find(e1) != poly_set.end() &&
                    std::find(fixed_poly.begin(), fixed_poly.end(), e1) == fixed_poly.end()) {
                    fixed_poly.push_back(e1);
                    break;
                }
            }
        }
        fixed_polys.push_back(fixed_poly);
    }
    polys = fixed_polys;
}

void import_obj(const char* fname, Verts3D& verts, Faces& polys) {
    verts.clear();
    polys.clear();
    std::ifstream fin(fname);
    std::string line;
    while (std::getline(fin, line)) {
        ltrim(line);
        rtrim(line);
        std::vector<std::string> strs = split(line, ' ');
        if (strs[0] == "v") {
            const Vector3f vert((float)std::stod(strs[1]), (float)std::stod(strs[2]), (float)std::stod(strs[3]));
            verts.push_back(vert);
        } else if (strs[0] == "f") {
            Face face;
            for (size_t i = 1; i < strs.size(); ++i) {
                const std::string faceStr = strs[i];
                size_t slash_pos = faceStr.find('/', 0);
                if (slash_pos == std::string::npos) {
                    face.push_back(std::stoi(faceStr) - 1);
                } else {
                    face.push_back(std::stoi(faceStr.substr(0, slash_pos)) - 1);
                }
            }
            polys.push_back(face);
        }
    }
}

void export_obj(const char* fname, const Verts3D& verts, const Faces& polys) {
    std::ofstream fout(fname);
    for (Vector3f vert : verts) {
        fout << "v " << std::setprecision(12) << (double)vert.x() << " " << (double)vert.y() << " " << (double)vert.z() << "\n";
    }
    for (Face poly : polys) {
        fout << "f";
        for (int p : poly) {
          fout << " " << (p+1);
        }
        fout << "\n";
    }
}

template<class VTYPE>
inline float pt_to_line_dist_sq(const VTYPE& pt, const VTYPE& a, const VTYPE& b) {
    const VTYPE p = pt - a;
    const VTYPE d = b - a;
    const float t = p.dot(d) / d.squaredNorm();
    const VTYPE v = p - d * std::min(std::max(t, 0.0f), 1.0f);
    return v.squaredNorm();
}

void line_line_intersection(const Vector3f& a1, const Vector3f& a2, const Vector3f& b1, const Vector3f& b2, Vector3f& pa, Vector3f& pb) {
  const Vector3f p13 = a1 - b1;
  const Vector3f p43 = b2 - b1;
  const Vector3f p21 = a2 - a1;
  const float d1343 = p13.dot(p43);
  const float d4321 = p43.dot(p21);
  const float d1321 = p13.dot(p21);
  const float d4343 = p43.dot(p43);
  const float d2121 = p21.dot(p21);
  const float denom = d2121 * d4343 - d4321 * d4321;
  const float numer = d1343 * d4321 - d1321 * d4343;
  const float mua = numer / denom;
  const float mub = (d1343 + d4321 * mua) / d4343;
  pa = a1 + mua * p21;
  pb = b1 + mub * p43;
}

void make_2d_projection(const Verts3D v3ds, const Face& poly, const Plane& plane, Verts2D& v2ds) {
    Matrix3f basis; Vector3f p;
    make_2d_projection(v3ds, poly, plane, v2ds, basis, p);
}

void make_2d_projection(const Verts3D v3ds, const Face& poly, const Plane& plane, Verts2D& v2ds, Matrix3f& basis, Vector3f& p) {
    //Find basis for plane x,y,n
    const Vector3f& n = plane.n;
    const Vector3f x = (v3ds[poly[1]] - v3ds[poly[0]]).normalized();
    const Vector3f y = n.cross(x);
    basis.transpose() << x, y, n;
    p = n * plane.d;

    //Create an array of projected vertices
    const size_t num = poly.size();
    v2ds.resize(num);
    for (size_t i = 0; i < num; ++i) {
        v2ds[i] = (basis * (v3ds[poly[i]] - p)).head<2>();
    }
}

Plane get_plane(const Verts3D& pts, const Face& poly) {
    Vector3d mean = Vector3d::Zero();
    for (int ix : poly) {
        mean += pts[ix].cast<double>();
    }
    mean /= double(poly.size());
    Matrix3d xx = Matrix3d::Zero();
    for (int ix : poly) {
        const Vector3d x = pts[ix].cast<double>() - mean;
        xx += x * x.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(xx, Eigen::ComputeFullU);
    const Vector3d n = svd.matrixU().col(2);
    return Plane(n.cast<float>(), (float)n.dot(mean));
}

Vector3f plane_intersection(const Plane& p1, const Plane& p2, const Plane& p3) {
    Matrix3f m; m.transpose() << p1.n, p2.n, p3.n;
    Vector3f d(p1.d, p2.d, p3.d);
    return m.inverse() * d;
}

void spread_planes(VectorXf& x, float spread, int iters) {
    const VectorXf orig(x);
    VectorXf delta = VectorXf::Zero(x.size());
    for (int iter = 0; iter < iters; ++iter) {
        delta.setZero();
        float max_dot = 0.0f;
        for (int i = 0; i < x.size(); i += 3) {
            const Eigen::Map<const Vector3f> p1(x.data() + i);
            const Vector3f n1 = p1.normalized();
            for (int j = 0; j < i; j += 3) {
                const Eigen::Map<const Vector3f> p2(x.data() + j);
                const Vector3f n2 = p2.normalized();
                const float dp = n1.dot(n2);
                max_dot = std::max(max_dot, std::abs(dp));
                if (dp >= 0.0) {
                    const Vector3f df = (n1 - n2).normalized();
                    Eigen::Map<Vector3f>(delta.data() + i) += df * dp;
                    Eigen::Map<Vector3f>(delta.data() + j) -= df * dp;
                } else {
                    const Vector3f df = (n1 + n2).normalized();
                    Eigen::Map<Vector3f>(delta.data() + i) += df * -dp;
                    Eigen::Map<Vector3f>(delta.data() + j) += df * -dp;
                }
            }
        }
        x += delta * spread;
        for (int i = 0; i < x.size(); i += 3) {
            Eigen::Map<Vector3f>(x.data() + i).normalize();
        }
        std::cout << max_dot << std::endl;
    }
    for (int i = 0; i < x.size(); i += 3) {
        Eigen::Map<Vector3f>(x.data() + i) *= Eigen::Map<const Vector3f>(orig.data() + i).norm();
    }
}

void y_to_v3ds(const VectorXf& y, Verts3D& verts) {
  verts.clear();
  for (int i = 0; i < y.size(); i += 3) {
    const Eigen::Map<const Vector3f> sub_x(y.data() + i);
    verts.emplace_back(sub_x);
  }
}

void v3ds_to_y(const Verts3D& verts, VectorXf& y) {
  y.resize(verts.size() * 3);
  for (size_t i = 0; i < verts.size(); ++i) {
    const Vector3f& v = verts[i];
    y[i * 3 + 0] = v.x();
    y[i * 3 + 1] = v.y();
    y[i * 3 + 2] = v.z();
  }
}

void x_to_planes(const VectorXf& x, Planes& planes) {
    planes.clear();
    for (int i = 0; i < x.size(); i += 3) {
        const Eigen::Map<const Vector3f> sub_x(x.data() + i);
        planes.emplace_back(sub_x);
    }
}

void planes_to_x(const Planes& planes, VectorXf& x) {
    x.resize(planes.size() * 3);
    for (size_t i = 0; i < planes.size(); ++i) {
        const Vector3f nd = planes[i].n * planes[i].d;
        x[i*3 + 0] = nd.x();
        x[i*3 + 1] = nd.y();
        x[i*3 + 2] = nd.z();
    }
}

void planes_to_v3ds(const Faces& dual_tris, const Planes& planes, Verts3D& verts) {
    verts.clear();
    for (const Face& face : dual_tris) {
        const Plane& plane_a = planes[face[0]];
        const Plane& plane_b = planes[face[1]];
        const Plane& plane_c = planes[face[2]];
        verts.push_back(plane_intersection(plane_a, plane_b, plane_c));
    }
}

void v3ds_to_planes(const Verts3D& pts, const Faces& polys, Planes& planes) {
    planes.clear();
    if (polys[0].size() == 3) {
        for (const Face& poly : polys) {
            planes.emplace_back(pts[poly[0]], pts[poly[1]], pts[poly[2]]);
        }
    } else {
        for (const Face& poly : polys) {
            planes.push_back(get_plane(pts, poly));
        }
    }
}

void x_to_v3ds(const VectorXf& x, const Faces& tris, Verts3D& verts) {
    Planes planes;
    x_to_planes(x, planes);
    planes_to_v3ds(tris, planes, verts);
}

void v3ds_to_x(const Verts3D& pts, const Faces& polys, VectorXf& x) {
    Planes planes;
    v3ds_to_planes(pts, polys, planes);
    planes_to_x(planes, x);
}

void y_to_x(const VectorXf& n, const VectorXf& y, VectorXf& x) {
    x.resize(n.size());
    for (int i = 0; i < y.size(); ++i) {
        const Eigen::Map<const Vector3f> sub_n(n.data() + i*3);
        Eigen::Map<Vector3f> sub_x(x.data() + i*3);
        const float mag = (std::abs(y[i]) > 1e-3f ? y[i] : 1e-3f);
        sub_x = sub_n.normalized() * mag;
    }
}

inline float cp_test(const Vector2f& p1, const Vector2f& p2, const Vector2f& p3) {
    return (p2.x() - p1.x()) * (p3.y() - p1.y()) -
           (p2.y() - p1.y()) * (p3.x() - p1.x());
}

bool point_in_polygon(const Vector2f& p, const Verts2D& pts, int& onEdge) {
    static const float epsilon = 1e-6f;
    static const float epsilon2 = 1e-5f;
    onEdge = -1;
    int windingNumber = 0;
    const Vector2f* p2 = &pts[pts.size() - 1];
    for (size_t i = 0; i < pts.size(); i++) {
        const Vector2f* p1 = &pts[i];
        const float d2 = pt_to_line_dist_sq(p, *p1, *p2);
        if (d2 < epsilon && (*p1 - p).squaredNorm() > epsilon2 && (*p2 - p).squaredNorm() > epsilon2) {
            onEdge = (int)i;
            return true;
        }
        if (p1->y() <= p.y()) {
            if (p2->y() > p.y() && cp_test(*p1, *p2, p) > 0.0f) {
                windingNumber++;
            }
        } else {
            if (p2->y() <= p.y() && cp_test(*p1, *p2, p) < 0.0f) {
                windingNumber--;
            }
        }
        p2 = p1;
    }
    return std::abs(windingNumber) == 1;
}

int count_crossings(const Verts3D& v3ds, const Plane& plane, const Face& poly) {
    static const float epsilon = 1e-3f;
    static const float epsilon2 = 1e-10f;
    static Verts2D v2ds;

    //Initialize results
    int crossings = 0;

    //Create an array of projected vertices
    make_2d_projection(v3ds, poly, plane, v2ds);

    //Iterate over all the line segments in order
    const size_t num = poly.size();
    Vector2f a1 = v2ds[0];
    for (size_t i = 1; i < num; ++i) {
        //Get the first line segment
        const Vector2f& a2 = v2ds[i];
        const Vector2f da = a2 - a1;

        Vector2f b1 = v2ds[num - 1];
        for (size_t j = 0; j < i - 1; ++j) {
            //Ignore edge case that should not intersect
            if (i - j == num - 1) {
                b1 = v2ds[j];
                continue;
            }

            //Get the other line segment
            const Vector2f& b2 = v2ds[j];
            const Vector2f db = b1 - b2;

            //Check if the lines intersect
            const Vector2f ba = b1 - a1;
            const float det = da.x()*db.y() - da.y()*db.x();
            if (det * det < epsilon2) {
                //Lines are parallel. Count as crossing if the distance between is too small.
                const float d = (ba - da * (ba.dot(da) / da.squaredNorm())).squaredNorm();
                if (d < epsilon) {
                    crossings += 1;
                }
            } else {
                //Lines are not parallel. Look for the intersection point if it exists.
                const float t = (db.y()*ba.x() - db.x()*ba.y()) / det;
                if (t > -epsilon && t < 1.0f + epsilon) {
                    const float s = (da.x()*ba.y() - da.y()*ba.x()) / det;
                    if (s > -epsilon && s < 1.0f + epsilon) {
                        crossings += 1;
                    }
                }
            }
            b1 = b2;
        }
        a1 = a2;
    }
    return crossings;
}

int count_crossings(const Verts3D& v3ds, const Planes& planes) {
    int crossings = 0;
    for (size_t i = 0; i < g_polys.size(); ++i) {
        crossings += count_crossings(v3ds, planes[i], g_polys[i]);
    }
    return crossings;
}

int count_intersections(const Verts3D& v3ds, const Planes& planes, const Plane& plane, const Face& poly, const Edges& other_edges) {
    static Verts2D v2ds;

    //Initialize results
    int intersections = 0;

    //Create an array of projected vertices
    Matrix3f basis; Vector3f p;
    make_2d_projection(v3ds, poly, plane, v2ds, basis, p);

    //Iterate over all valid edges
    for (Edge edge : other_edges) {
        const int ex1 = edge.first;
        const int ex2 = edge.second;

        //Get edge vertices
        const Vector3f& p1 = v3ds[ex1];
        const Vector3f& p2 = v3ds[ex2];

        //Get plane intersection with edge
        Vector3f in3d;
        if (!plane.intersect(p1, p2, in3d)) continue;
        const Vector2f in2d = (basis * (in3d - p)).head<2>();

        //Check if plane point is inside the polygon
        int edgeIx = -1;
        if (point_in_polygon(in2d, v2ds, edgeIx)) {
            if (edgeIx >= 0 && poly.size() > 3) {
                //If edges are already crossed, don't double count this intersection
                const int px1 = poly[edgeIx];
                const int px2 = poly[(edgeIx + poly.size() - 1) % poly.size()];
                Vector3f pa, pb;
                line_line_intersection(v3ds[ex1], v3ds[ex2], v3ds[px1], v3ds[px2], pa, pb);
                if ((pa - pb).squaredNorm() < 1e-6f) {
                    //The edges intersect, but are they both part of the same crossed face?
                    //Get the plane and compare it with others
                    const Plane intersection_plane(v3ds[ex1], v3ds[ex2], v3ds[px1]);
                    const Vector3f intersection_nd = intersection_plane.n * intersection_plane.d;
                    static const float MIN_DIST_SQ = 1e-8f;
                    float dist = 999.0f;
                    for (const Plane& test_plane : planes) {
                        dist = (intersection_nd - (test_plane.n * test_plane.d)).squaredNorm();
                        if (dist < MIN_DIST_SQ) { break; }
                    }

                    //Matches an existing plane, this must actually be a crossing, not intersection.
                    if (dist < MIN_DIST_SQ) {
                        continue;
                    }
                }
            }
            intersections += 1;
            //return 1;
        }
    }
    return intersections;
}

int count_intersections(const Verts3D& v3ds, const Planes& planes) {
    //Create a quick-access set for the polygon
    static int poly_set_topology = -1;
    static std::vector<Edges> poly_sets;
    if (g_topology != poly_set_topology) {
        poly_sets.resize(g_polys.size());
        for (size_t i = 0; i < g_polys.size(); ++i) {
            poly_sets[i].clear();
            const std::vector<int>& poly = g_polys[i];
            std::unordered_set<int> poly_set(poly.begin(), poly.end());
            for (const Edge& edge : g_edges) {
                const int ex1 = edge.first;
                const int ex2 = edge.second;
                if (poly_set.count(ex1) == 0 && poly_set.count(ex2) == 0) {
                    poly_sets[i].push_back(edge);
                }
            }
        }
        poly_set_topology = g_topology;
    }
    int intersections = 0;
    for (size_t i = 0; i < g_polys.size(); ++i) {
        intersections += count_intersections(v3ds, planes, planes[i], g_polys[i], poly_sets[i]);
    }
    return intersections;
}

float angle_penalty(const Verts3D& verts) {
    float mdp = 0.0f;
    const size_t n_sides = g_polys[0].size();
    for (const Face& poly : g_polys) {
        for (size_t i = 0; i < n_sides; ++i) {
            const Vector3f p1 = verts[poly[i]];
            const Vector3f p2 = verts[poly[(i+1) % n_sides]];
            const Vector3f p3 = verts[poly[(i+2) % n_sides]];
            const Vector3f d1 = p1 - p2;
            const Vector3f d2 = p3 - p2;
            const float mags = d1.norm() * d2.norm();
            mdp = std::max(mdp, abs(d1.dot(d2) / mags));
        }
    }
    return mdp;
}

float dist_penalty(const Verts3D& verts) {
    float min_dist_sq = 99999.0f;
    float max_dist_sq = 0.0f;
    for (size_t i = 0; i < verts.size(); ++i) {
        const Vector3f& p = verts[i];
        for (const Edge& edge : g_edges) {
            if (edge.first == i || edge.second == i) { continue; }
            const Vector3f& a = verts[edge.first];
            const Vector3f& b = verts[edge.second];
            const float dist_sq = pt_to_line_dist_sq(p, a, b);
            min_dist_sq = std::min(min_dist_sq, dist_sq);
            max_dist_sq = std::max(max_dist_sq, dist_sq);
        } 
    }
    const float mratio = min_dist_sq / max_dist_sq;
    return 1.0f - std::sqrt(mratio);
}

float length_penalty(const Verts3D& verts) {
    float min_dist_sq = 99999.0f;
    float max_dist_sq = 0.0f;
    for (const Edge& edge : g_edges) {
        const Vector3f& a = verts[edge.first];
        const Vector3f& b = verts[edge.second];
        const float dist_sq = (a - b).squaredNorm();
        min_dist_sq = std::min(min_dist_sq, dist_sq);
        max_dist_sq = std::max(max_dist_sq, dist_sq);
    }
    return 1.0f - std::sqrt(min_dist_sq / max_dist_sq);
}

float plane_penalty(const Planes& planes) {
    float mdp = 0.0f;
    for (const Face& triLoop : g_tris) {
        const size_t n_sized = triLoop.size();
        for (size_t i = 0; i < n_sized; ++i) {
            const Plane& p1 = planes[triLoop[i]];
            const Plane& p2 = planes[triLoop[(i + 1) % n_sized]];
            mdp = std::max(mdp, abs(p1.n.dot(p2.n)));
        }
    }
    return mdp;
}

float q_penalty(const Verts3D& verts) {
  const float dp = dist_penalty(verts);
  const float lp = length_penalty(verts);
  const float ap = angle_penalty(verts);
  return std::max(ap, std::max(dp, lp));
}

float objective_cross_int_qlim(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    if (angle_penalty(v3ds) > 0.999f || dist_penalty(v3ds) > 0.999f) {
        return 99999.0f;
    }
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_crossings * 100 + num_intersections);
}

float objective_cross_int_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_crossings * 100 + num_intersections) + q;
}

float objective_cross_zint(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    if (num_crossings == 0) {
        const int num_intersections = count_intersections(v3ds, planes);
        return float(num_intersections);
    } else {
        return float(num_crossings * 100);
    }
}

float objective_cross_int(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_crossings * 100 + num_intersections);
}

float objective_cross(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    return float(num_crossings);
}

float objective_int_cross_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections * 100 + num_crossings) + q;
}

float objective_int_cross(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections * 100 + num_crossings);
}

float objective_int_zcross_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_intersections = count_intersections(v3ds, planes);
    if (num_intersections == 0) {
        const int num_crossings = count_crossings(v3ds, planes);
        return float(num_crossings) + q;
    } else {
        return float(num_intersections * 100);
    }
}

float objective_int_zcross(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_intersections = count_intersections(v3ds, planes);
    if (num_intersections == 0) {
        const int num_crossings = count_crossings(v3ds, planes);
        return float(num_crossings);
    } else {
        return float(num_intersections * 100);
    }
}

float objective_int_qlim(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    if (angle_penalty(v3ds) > 0.999f || dist_penalty(v3ds) > 0.999f) {
        return 99999.0f;
    }
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections);
}

float objective_int_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections) + q;
}

float objective_int(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections);
}

float objective_sum_qlim(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    if (angle_penalty(v3ds) > 0.999f || dist_penalty(v3ds) > 0.999f) {
        return 99999.0f;
    }
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections + num_crossings);
}

float objective_sum_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections + num_crossings) + q;
}

float objective_sum(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections + num_crossings);
}

float objective_wsum(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections + num_crossings * 2);
}

float objective_wsum2(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections * 2 + num_crossings);
}

float objective_wsum_q(const VectorXf& x) {
    static Planes planes;
    static Verts3D v3ds;
    x_to_planes(x, planes);
    planes_to_v3ds(g_tris, planes, v3ds);
    const float q = q_penalty(v3ds);
    const int num_crossings = count_crossings(v3ds, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections + num_crossings * 2) + q;
}

float objective_dual(const VectorXf& y) {
    static Planes planes;
    static Verts3D v3ds;
    y_to_v3ds(y, v3ds);
    if (angle_penalty(v3ds) > 0.9995f || dist_penalty(v3ds) > 0.9995f) {
        return 99999.0f;
    }
    v3ds_to_planes(v3ds, g_polys, planes);
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections);
}

float objective_dual_q(const VectorXf& y) {
    static Planes planes;
    static Verts3D v3ds;
    y_to_v3ds(y, v3ds);
    v3ds_to_planes(v3ds, g_polys, planes);
    const float q = std::max(q_penalty(v3ds), plane_penalty(planes));
    const int num_intersections = count_intersections(v3ds, planes);
    return float(num_intersections) + q;
}