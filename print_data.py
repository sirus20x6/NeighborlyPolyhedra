import os

num_topologies = 59
fstr = "{:<9}{:<10}{:<9}{:<9}{:<8}"

print(fstr.format("Topology", "Solutions", "BestZInt", "BestCrsZ", "BestSum"))
best_results = [None, None, None]
for i in range(num_topologies):
    min_zint = 999999
    min_crossz = 999999
    min_sum = 999999
    num_files = 0
    for fname in os.listdir("topology_" + str(i)):
        if not fname.endswith('.obj'): continue
        fields = fname.split('.')[0].split('_')
        crossings = int(fields[1][1:])
        intersections = int(fields[2][1:])
        ci_sum = crossings + intersections
        if crossings == 0:
            min_zint = min(min_zint, intersections)
        if intersections == 0:
            min_crossz = min(min_crossz, crossings)
        min_sum = min(min_sum, ci_sum)
        if best_results[0] is None or min_zint < best_results[0][0]:
            best_results[0] = (min_zint, "topology_" + str(i) + "/" + fname)
        if best_results[1] is None or min_crossz < best_results[1][0]:
            best_results[1] = (min_crossz, "topology_" + str(i) + "/" + fname)
        if best_results[2] is None or ci_sum < best_results[2][0]:
            best_results[2] = (ci_sum, "topology_" + str(i) + "/" + fname)
        num_files += 1
    if num_files > 0:
        if min_zint > 99: min_zint = '-'
        if min_crossz > 99: min_crossz = '-'
        print(fstr.format(i, num_files, min_zint, min_crossz, min_sum))
    else:
        print(fstr.format(i, num_files, '-', '-', '-'))
print("===============================")
print("Best Zero-Int:   " + best_results[0][1])
print("Best Cross-Zero: " + best_results[1][1])
print("Best Sum:        " + best_results[2][1])