from bisect import bisect
from collections import defaultdict
from itertools import combinations
from math import floor, sqrt

import trimesh
from progress.bar import Bar

PADDING = 0.05
VERTICAL_SCALE = sqrt(2) / 4  # Vertical scale. Constant a from paper.

def normalise(mesh):
    """Rescale and translate the input mesh so that it is inside the unit cube.
    Return the new mesh along with the scale and translation needed to undo this transformation."""

    mins, maxes = mesh.bounds

    scale = (1 - 2 * PADDING) / max(i - j for i, j in zip(maxes, mins))
    translate = [-i * scale + PADDING for i in mins]

    return affine(mesh, scale, translate), 1 / scale, [x - PADDING / scale for x in mins]

def affine(mesh, scale, translate):
    """Rescale and translate a mesh."""

    mesh = mesh.copy()
    mesh.apply_transform(trimesh.transformations.scale_and_translate(scale, translate))

    return mesh

def midnormal(N, mesh):
    """Return the mesh obtained from middle normal disks."""

    # Using mesh.contains([P]) is incredibly slow.
    # So we will take advantage of the fact that our vertices are in vertical stacks.

    dx = 1 / N
    dy = dx * sqrt(3) / 2
    dz = VERTICAL_SCALE * dx

    heights = defaultdict(list)  # Mapping lattice point (i, j) |--> list of heights of triangles of mesh above (i, j, 0).
    tetrahedra = set()  # The set of tetrahedra that could contain a normal disk.

    with Bar('Evaluating triangles', max=len(mesh.triangles), width=80, suffix='%(percent)d%%  ETA: %(eta)ds') as bar:
        for index, triangle in enumerate(mesh.triangles):
            if index % 1000 == 0:
                bar.next(1000)

            # Find the bounding box of this triangle.
            x_min, y_min, z_min = triangle.min(axis=0)
            x_max, y_max, z_max = triangle.max(axis=0)

            i_min, i_max = floor(x_min / dx), floor(x_max / dx) + 2
            j_min, j_max = floor(y_min / dy), floor(y_max / dy) + 1

            (Nx, Ny, Nz), K = spanning_plane(*triangle)

            # Compute z coordinate where the vertical line above (x0, y0) meets the plane spanned by v0, v1, v2.
            # Eqn of plane is then N . X == K
            # So rearrange Nx * x0 + Ny * y0 + Nz * z == K, to solve for z.

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    # Define the four vertices of the parallelogram:
                    V0 = (i + 0, j + j % 2)
                    V1 = (i + 1, j + j % 2)
                    V2 = (i + 1/2, j + 1 - j % 2)
                    V3 = (i - 1/2, j + 1 - j % 2)

                    corner_heights = [(K - Nx * x * dx - Ny * y * dy) / Nz for x, y in [V0, V1, V2, V3]] if Nz else [-1, 1]
                    corner_min, corner_max = max(min(corner_heights), z_min), min(max(corner_heights), z_max)
                    k_min, k_max = floor(corner_min / dz) - 4, floor(corner_max / dz) + 1

                    for V in [(V0, V1, V2), (V0, V3, V2)]:
                        for k in range(k_min, k_max):
                            v0, v1, v2 = V[k % 3:] + V[:k % 3]
                            A = (v0[0], v0[1], k + 0 + i % 3)
                            B = (v1[0], v1[1], k + 1 + i % 3)
                            C = (v2[0], v2[1], k + 2 + i % 3)
                            D = (v0[0], v0[1], k + 3 + i % 3)
                            tetrahedra.add((A, B, C, D))

                    if Nz:
                        x, y = (i - (j % 2) / 2) * dx, j * dy
                        if point_is_under_triangle(x, y, *triangle):
                            heights[i - (j % 2) / 2, j].append((K - Nx * x - Ny * y) / Nz)

    # Sort the heights above each point.
    for above in heights.values():
        # Lines must intersect the surface an even number of times.
        assert len(above) % 2 == 0
        above.sort()

    empty = 0  # Count the number of tetrahedra that don't contain any normal disks.
    vertices = []
    print(f"Tiling unit cube with {len(tetrahedra)} tetrahedra")
    with Bar('Evaluating tetrahedra', max=len(tetrahedra), width=80, suffix='%(percent)d%%  ETA: %(eta)ds') as bar:
        for index, (a, b, c, d) in enumerate(tetrahedra):  # Note the quads depend on this ordering.
            if index % 1000 == 0:
                bar.next(1000)

            A = (a[0] * dx, a[1] * dy, a[2] * dz)
            B = (b[0] * dx, b[1] * dy, b[2] * dz)
            C = (c[0] * dx, c[1] * dy, c[2] * dz)
            D = (d[0] * dx, d[1] * dy, d[2] * dz)

            # Find Euclidean coordinates for edge midpoints
            AB, AC, AD, BC, BD, CD = [midpoint(p, q) for p, q in combinations([A, B, C, D], r=2)]

            signs = [((len(heights[i, j]) - bisect(heights[i, j], k * dz)) % 2 == 1 if (i, j) in heights else False) for i, j, k in [a, b, c, d]]
            if not signs[0]:  # Flip so sign fA is True.
                signs = [not sign for sign in signs]

            if signs == [True, True, True, True]:  # No disk.
                empty += 1
            elif signs == [True, False, False, False]:  # Triangle near vertex A.
                vertices.extend([AB, AC, AD])
            elif signs == [True, False, True, True]:  # Triangle near vertex B.
                vertices.extend([AB, BC, BD])
            elif signs == [True, True, False, True]:  # Triangle near vertex C.
                vertices.extend([AC, BC, CD])
            elif signs == [True, True, True, False]:  # Triangle near vertex D.
                vertices.extend([AD, BD, CD])
            elif signs == [True, False, False, True]:  # Quadrilateral separating AD & BC.
                vertices.extend([AB, BD, AC])
                vertices.extend([BD, AC, CD])
            elif signs == [True, True, False, False]:  # Quadrilateral separating AB & CD
                vertices.extend([AD, BD, AC])
                vertices.extend([BD, AC, BC])
            elif signs == [True, False, True, False]:  # Quadrilateral separating AC & BD. Since VERTICAL_SCALE <= sqrt(2) / 4, we need the AD--BC diagonal.
                vertices.extend([AB, AD, BC])
                vertices.extend([AD, BC, CD])

    faces = [[i, i+1, i+2] for i in range(0, len(vertices), 3)]
    print(f"Empty tetrahedra {empty}")

    mesh = trimesh.Trimesh(vertices, faces)

    # assert mesh.is_watertight
    return mesh

def remove_quads(mesh):
    """Return a new mesh where degree 4 vertices have been removed."""

    vertices = mesh.vertices
    faces = mesh.faces

    # A list where the ith entry contains the set of vertices that are adjacent in the 1-skeleton to the ith vertex.
    neighbours = [set(x) for x in mesh.vertex_neighbors]
    valences = [len(x) for x in neighbours]

    new_triangles = [face for face in faces if all(valences[v_index] != 4 for v_index in face)]

    # For a face that is a quad, identify its good diagonal.
    # This is the one not between two valence 6 vertices
    for index in range(len(vertices)):
        if valences[index] == 4:
            N = neighbours[index]
            a = next(iter(N))  # Get one of the neighbours.
            b, c = N.intersection(neighbours[a])  # Get the two neighbours adjacent to a.
            [d] = N.difference({a, b, c})  # Get the final neighbor.
            if valences[a] + valences[d] > valences[b] + valences[c]:
                new_triangles.extend([[a, b, c], [b, c, d]])  # Diagonal b--c.
            else:
                new_triangles.extend([[b, a, d], [a, d, c]])  # Diagonal a--d.

    return trimesh.Trimesh(vertices, new_triangles)

def displace_to_surface(surface, original):
    """Return a new mesh by pushing the vertices of surface to the closest point of original."""

    surface = surface.copy()
    surface.vertices, _, _ = original.nearest.on_surface(surface.vertices)

    return surface

### Some boring functions for basic Euclidean geometry.

def midpoint(p1, p2):
    return tuple(0.5 * a + 0.5 * b for a, b in zip(p1, p2))

def point_is_under_triangle(x, y, p0, p1, p2):
    """Determine whether the vertical line through (x, y, 0) intersects the triangle (p0, p1, p2)."""

    dX = x - p2[0]
    dY = y - p2[1]
    dX21 = p2[0] - p1[0]
    dY12 = p1[1] - p2[1]
    # The Barycentric coordinates of (x, y) wrt the triangle.
    s = dY12 * dX + dX21 * dY
    t = (p2[1] - p0[1]) * dX + (p0[0] - p2[0]) * dY
    D = dY12 * (p0[0] - p2[0]) + dX21 * (p0[1] - p2[1])

    if D < 0:
        return s <= 0 and t <= 0 and s + t >= D

    return s >= 0 and t >= 0 and s + t <= D

def spanning_plane(v0, v1, v2):
    """Return the normal N and offset K of the plane containing v0, v1 and v2.

    These are so that X = (x, y, z) is on the plane if and only if N . X == K"""

    Nx = (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1])
    Ny = (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2])
    Nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
    K = Nx * v0[0] + Ny * v0[1] + Nz * v0[2]
    return (Nx, Ny, Nz), K
