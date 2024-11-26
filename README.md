# NeighborlyPolyhedra
This is the culmination of all my research in answering the question; Is there another polyhedron besides the Tetrahedron and Szilassi Polyhedron where all faces share an edge with all other faces?

Video: [YouTube](https://youtu.be/5dd8_N_nKRI).

The conclusion is: Sort of. I believe there isn't a shape that is _completely_ intersection-free, but there IS a minimally intersecting one, a near-miss, that can be considered the _best-possible_ or _closest_ solution. I've named this shape the "Razorcross" and it's documented here in this repository, as well as all the supporting code to generate all the neighborly polyhedra. Note that there is no rigorous proof about the optimality of this solution, this conclusion has been made based on my observations of patterns, statistical analysis, and brute-forcing of the problem. If there was a better solution, I almost certainly would have found it by now.

## Running The Code
To start a search, run `main.cpp`.

### Requirements
* C++17 compatible compiler (I'm using MSVC 2019).
* [Eigen-3.4.0](https://eigen.tuxfamily.org/) though similar versions should work as well.
* [Cairo-1.17.2](https://www.cairographics.org/) (optional) this is only needed to render paper cutouts.

### Basics
Running the main code will ask for a seed number. It will then start a search, taking turns through each of the 59 topologies sequentially, looking for solutions. To run multiple threads, I simply launch the executable in different processes with different seeds. This is the lazy way to do multi-threading, but it works! This code is programmed to only save solutions if it's good enough to be notable. By that, I mean it either has no crossings (all simple polygons) or 10 or fewer intersections. The solutions get saved into their own topology folders. I've provided the top solutions I've found for each topology already. I wouldn't be surprised if you can find a solution that lowers the record for some topologies, but I've studied all the low intersection, most consistent, most symmetric, and best looking models more extensively, and I doubt there would be further improvements there.

Solutions get saved as `shape_c#_i#_#.obj` where the first number is the number of crossings, the second number is the number of edge-face intersections, and the third number is the current iteration of the solver when it found the solution.

### Advanced
There's a ton of utility and functionality I added during the research that you can use, but you'll need to add it in and recompile. Here are some common ones:
* **Hyperparameters** You can adjust the solver parameters `max_iters`, `clusters`, `beta`, and `sigma`.
* **Objective Function** You can change the objective function by replacing `objective_sum` with any of the other objective functions listed at the end of `util.h`. Note that when the objective function changes, you'll usually also need to change the early exit conditions in `solver.cpp` since the cost function units may be different.
* **Symmetry** To add symmetry enforcement, there is a boolean argument to the solver `use_symmetry`. There are different symmetries you can use at the top of `solver.cpp`. These symmetries are for specific topologies, usually 6, 37, 42, 49, 55, and 58.
* **Single Topology** If you'd like to focus on one topology instead of equal computation to all, just modify `g_topology = your_number_here`;

### The Dual Problem
The problem of finding a polyhedra with neighborly faces also has a dual problem, which is to find a polyhedron with neighborly vertices. In terms of the Szilassi polyhedron, the dual problem is analogous to finding the [Császár polyhedron](https://en.wikipedia.org/wiki/Cs%C3%A1sz%C3%A1r_polyhedron). The dual polyhedron has the same number of holes and edges, but with the number of faces and vertices swapped, and all faces are triangles. Despite the similarity, and as far as I can tell from my research, having a solution or proving there is no solution to one problem does not answer the dual problem. In fact, the K12 polyhedron that is neighborly in its vertices was already proven impossible (see "Nonrealizable Minimal Vertex Triangulations of Surfaces" in the further reference section below for more details).

Since it was already proven, the dual problem was not as interesting to me, and wasn't mentioned in the video. Still, the impossibility proof does not attempt to answer the question of what is the minimum number of intersections the polyhedron could have and what does the shape look like? Since I already had all the code, there weren't many changes needed to run a similar solve, you can enable the `DUAL_PROBLEM` preprocessor definition and use one of the `objective_dual_*` objective functions if you'd like to try it. I didn't spend as much time on this but I did find a shape with 4 intersections in manifold 44. It's not beautiful or symmetric, but I included it in the folder `Dual44` if you're curious. It may be possible to find an order-2 symmetric solution in manifold 44 with 4 intersections, but I couldn't seem to get it under 6 when I forced the symmetry. It may also be possible to find other solutions with 2 or 3 intersections, but again I didn't spend much time on this problem.

### Quality
Once you find a solution, you may want to improve the quality to get a better looking solution. This is done by adding a 'quality' penalty to the objective function, which is any objective function that has a `q` in it. This may make it harder to converge on a solution in the search, and you may not have had enough iterations to fully converge anyway. So what I usually do is load the saved model and run the `study_sample` function to improve it and converge on the highest quality shape. This means specifically:
* Opening sharp angles (near 0 degrees).
* Closing open angles (near 180 degrees).
* Making sure edge lengths are not relatively too small or large.
* Adding more clearance in the polygons so they're not 'almost' crossing.

The results are saved in the main directory as `study_c#_i#_#.obj`

## The Razorcross
All files related to the Razorcross are found in the `Razorcross` folder.
* `original_polygons.obj` The 12 polygons straight from the solver. Due to the intersections, this will not be a proper manifold.
* `triangulated_manifold.obj` A triangulated version that adds edges at the intersections. A proper manifold that can be 3D printed.
* `printable_magnets_half1.stl` First half of the model with holes for 4.8mm diameter magnets and in a better printing position.
* `printable_magnets_half2.stl` Second half of the model above. It's exactly a mirror image.
* `texture.png` Texture to use with the uv coordinates of `original_polygons.obj` to color the 12 sides.
* `cuttout.png` The 3 unique faces of the Razorcross. You would need 2 sheets plus 2 mirrored sheets to get all 12 faces.
* `edge_graph.dot` The edge graph of manifold-42.

## General Observations
Shapes with more symmetry tend to also be the ones with the fewest intersections. Below are some examples that have "180 degree rotation" and/or "point reflection" symmetry. These highly symmetric shapes only use 3 or 6 unique faces and their mirrors.
* **topology_6**
  * 0 crossings, 8 intersections (shape_c0_i8_0.obj)
  * 10 crossings, 0 intersections (shape_c10_i0_2.obj)
* **topology_37**
  * 0 crossings, 8 intersections (study_c0_i8_bestlooking.obj)
  * 4 crossings, 0 intersections (shape_c4_i0_0.obj)
* **topology_42 (Razorcross)**
  * 0 crossings, 4 intersections (shape_c0_i4_optimalsymmetric.obj)
  * 4 crossings, 0 intersections (shape_c4_i0_4.obj)
* **topology_49**
  * 0 crossings, 8 intersections (shape_c0_i8_5.obj)
  * 8 crossings, 0 intersections (shape_c8_i0_0.obj))
* **topology_55**
  * 0 crossings, 16 intersections (shape_c0_i16_0.obj)
  * 10 crossings, 0 intersections (shape_c10_i0_6.obj)
* **topology_58**
  * 0 crossings, 10 intersections (study_c0_i10_28.obj)
  * 4 crossings, 0 intersections (shape_c4_i0_2.obj)

The paper about "Neighborly 2-Manifolds" listed below has a table of automorphism groups for the dual graph problem. These correlate with the neighborly problem, but only certain symmetry groups seem to have symmetries in the dual problem. Note that the paper 1-indexes the topologies whereas I always use 0-index, so you'll need to subtract 1 from the paper numbering scheme to match mine.

## Further reference
[Szilassi polyhedron](https://en.wikipedia.org/wiki/Szilassi_polyhedron)

[Neighborly 2-Manifolds with 12 Vertices](https://doi.org/10.1006/jcta.1996.0069)

[Nonrealizable Minimal Vertex Triangulations of Surfaces](https://arxiv.org/abs/0801.2582)

[sci.math newsletter chain](https://ics.uci.edu/~eppstein/junkyard/szilassi.html)