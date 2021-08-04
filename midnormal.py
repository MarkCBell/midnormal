
import argparse
from math import pi

import trimesh
import plotille

from mesh import normalise, affine, midnormal, remove_quads, displace_to_surface

def main():
    parser = argparse.ArgumentParser(description='Refine a triangulation')
    parser.add_argument('N', type=int, help='number of tetrahedra')
    parser.add_argument('path', type=str, help='path to .off file')
    parser.add_argument('--output', '-o', type=str, default='output.off', help='path to save output to')
    args = parser.parse_args()

    original_mesh = trimesh.load(args.path)
    # assert original_mesh.is_watertight

    print("Angles for original mesh")
    print_angle_stats(original_mesh)

    # Translate the original mesh inside the unit cube.
    # We save the scale and translation so we can move it back to it's original size later.
    unit_mesh, unscale, untranslate = normalise(original_mesh)

    midnormal_mesh = midnormal(args.N, unit_mesh)

    quadless_mesh = remove_quads(midnormal_mesh)

    # Get largest connected component.
    quadless_mesh = max(quadless_mesh.split(), key=lambda m: len(m.vertices))

    # Project the vertices back to the original mesh.
    projected_mesh = displace_to_surface(quadless_mesh, unit_mesh)

    # Rescale back to original size.
    rescalled_mesh = affine(projected_mesh, unscale, untranslate)

    if args.output:
        with open(args.output, "w") as output_off:
            output_off.write(trimesh.exchange.off.export_off(rescalled_mesh))

    print("Angles for projected mesh")
    print_angle_stats(projected_mesh)

##########################################################################

def print_angle_stats(mesh):
    angles = [angle * 180.0 / pi for vertex in trimesh.triangles.angles(mesh.triangles) for angle in vertex]

    print(plotille.histogram(angles, bins=180, X_label='angle', height=20, x_min=0, x_max=180))
    print(f"min_angle: {min(angles)}")
    print(f"max_angle: {max(angles)}")

if __name__ == '__main__':
    main()
