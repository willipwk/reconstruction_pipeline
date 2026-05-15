import argparse

import fast_simplification
import numpy as np
import open3d as o3d


def simplify_mesh(mesh_path: str, output_path: str, target_reduction: float):
    if not 0.0 <= target_reduction < 1.0:
        raise ValueError("target_reduction must be in [0.0, 1.0).")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    simplified_points, simplified_triangles = fast_simplification.simplify(points, triangles, target_reduction)
    simplified_mesh = o3d.geometry.TriangleMesh()
    simplified_mesh.vertices = o3d.utility.Vector3dVector(simplified_points)
    simplified_mesh.triangles = o3d.utility.Vector3iVector(simplified_triangles)
    o3d.io.write_triangle_mesh(output_path, simplified_mesh)


def main():
    parser = argparse.ArgumentParser(description="Simplify a triangle mesh.")
    parser.add_argument("input_path", help="Input mesh path.")
    parser.add_argument("output_path", help="Output simplified mesh path.")
    parser.add_argument(
        "--target-reduction",
        type=float,
        default=0.9,
        help="Fraction of triangles to remove.",
    )
    args = parser.parse_args()
    simplify_mesh(args.input_path, args.output_path, args.target_reduction)


if __name__ == "__main__":
    main()
