# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    :param v: Input vector to normalize.
    :param dtype: NumPy data type for the normalized vector.

    :return: Normalized vector with unit length.
    """
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def plane_from_quad_origin_last(
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a coordinate system from a quad with the last vertex as origin.

    Constructs a plane coordinate system (origin, u-vector, v-vector, normal)
    from four vertices where the last vertex serves as the origin.

    :param vertices: Array of shape (4, 3) containing the four vertices of the quad.

    :return: Tuple containing (origin, u-vector, v-vector, normal vector).

    :raises ValueError: If vertices is not a (4, 3) array.
    """
    if vertices.shape != (4, 3):
        raise ValueError("vertices must be a (4, 3) array")

    vertices = np.asarray(vertices)

    p0 = vertices[3]
    u = normalize(vertices[2] - p0)
    v = normalize(vertices[0] - p0)
    n = normalize(np.cross(u, v))
    return p0, u, v, n


def intersect_segment_plane(
    a: np.ndarray,
    b: np.ndarray,
    p0: np.ndarray,
    n: np.ndarray,
    *,
    eps_plane: float = 1e-5,
) -> np.ndarray | None:
    """
    Compute intersection of a line segment with a plane.

    Finds the intersection point between a line segment defined by points A and B
    and a plane defined by point P0 and normal vector n.

    :param a: Start point of the line segment.
    :param b: End point of the line segment.
    :param p0: Point on the plane.
    :param n: Normal vector of the plane.
    :param eps_plane: Tolerance for plane distance calculations.

    :return: Intersection point if it exists within the segment, None otherwise.
    """
    da = np.dot(a - p0, n)
    db = np.dot(b - p0, n)

    # Both on same side or both almost on plane → treat as no crossing
    if da * db > 0 or (abs(da) < eps_plane and abs(db) < eps_plane):
        return None

    denom = da - db
    if abs(denom) < eps_plane:
        return None

    t = da / denom
    if 0.0 <= t <= 1.0:
        return a + t * (b - a)
    return None


def clean_extent_for_intersection(
    extent: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """
    Clean and prepare extent array for 3D intersection calculations.

    :param extent: Input extent array, shape (2, 2) or (2, 3)
    :param vertices: Array of vertices to extract Z bounds from, shape (N, 3)

    :return: Cleaned extent array with shape (2, 3)
    """
    if not isinstance(extent, np.ndarray):
        raise TypeError("Expected a numpy array of extent values.")

    if not extent.ndim == 2 or extent.shape not in [(2, 3), (2, 2)]:
        raise TypeError("Expected a 2D numpy array with 2 points and 2 or 3 columns")

    if extent.shape[1] == 2 or np.all(extent[:, 2] == 0):
        z_coords = vertices[:, 2]
        zmin, zmax = float(z_coords.min()), float(z_coords.max())
        # always have 1 and -1 padding to avoid precision issues
        z_bounds = np.array([[zmin - 1], [zmax + 1]], dtype=np.float64)
        extent = np.column_stack([extent.astype(np.float64), z_bounds])

    return extent


#  pylint: disable=too-many-locals
def compute_plane_box_intersections(
    p0: np.ndarray,
    n: np.ndarray,
    extent: np.ndarray,
    eps_plane: float,
) -> np.ndarray:
    """
    Compute intersections between a plane and the edges of a 3D bounding box.

    Finds all points where the plane intersects with the 12 edges of a 3D box
    defined by its min/max coordinates.

    :param p0: Point on the plane.
    :param n: Normal vector of the plane.
    :param extent: coordinates of (xmin, xmax, ymin, ymax, zmin, zmax) defining the box.
    :param eps_plane: Tolerance for plane distance calculations.

    :return: Array of intersection points, shape (N, 3) where N is the number of intersections.
    """
    # Expect extent to be already cleaned (shape (2, 3))
    xmin, ymin, zmin = extent[0]
    xmax, ymax, zmax = extent[1]

    c = np.asarray(
        [
            [xmin, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmax, ymin, zmin],
            [xmax, ymin, zmax],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
        ],
        dtype=np.float64,
    )

    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (7, 6),
        (7, 5),
        (7, 3),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (4, 5),
        (4, 6),
    ]

    points: list[np.ndarray] = []
    for a, b in edges:
        p = intersect_segment_plane(c[a], c[b], p0, n, eps_plane=eps_plane)
        if p is not None:
            points.append(p)

    if not points:
        return np.zeros((0, 3), dtype=np.float64)

    return np.asarray(points, dtype=np.float64)


def project_to_uv(
    p: np.ndarray,
    p0: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Project a 3D point onto a 2D UV coordinate system.

    Projects a point P onto the UV plane defined by origin P0 and basis vectors u and v.

    :param p: 3D point to project.
    :param p0: Origin of the UV coordinate system.
    :param u: U-axis basis vector.
    :param v: V-axis basis vector.

    :return: 2D coordinates in the UV system.
    """
    d = np.asarray(p, dtype=np.float64) - p0
    return np.array([np.dot(d, u), np.dot(d, v)], dtype=np.float64)


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Ensure polygon vertices are ordered counter-clockwise.

    Reverses the vertex order if the polygon area is negative (clockwise orientation).

    :param poly: Array of shape (N, 2) containing polygon vertices.

    :return: Polygon vertices in counter-clockwise order.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    poly_area = 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    if poly_area < 0.0:
        return poly[::-1]

    return poly


def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """
    Test if a point is on the inside of a directed edge.

    Uses the cross product to determine if point p is on the left side
    (inside) of the directed edge from a to b.

    :param p: Point to test.
    :param a: Start point of the edge.
    :param b: End point of the edge.

    :return: True if the point is inside (left of) the edge, False otherwise.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0


def intersection_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps_collinear: float,
) -> np.ndarray:
    """
    Compute intersection point of two 2D line segments.

    Finds the intersection point of line segment p1-p2 with line segment a-b.
    If the segments are collinear, returns p2 as a fallback.

    :param p1: Start point of the first segment.
    :param p2: End point of the first segment.
    :param a: Start point of the second segment.
    :param b: End point of the second segment.
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Intersection point or p2 if segments are collinear.
    """
    a_vec = p2 - p1
    b_vec = b - a
    c_vec = a - p1
    denom = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
    if abs(denom) < eps_collinear:
        return p2
    t = (c_vec[0] * b_vec[1] - c_vec[1] * b_vec[0]) / denom
    return p1 + t * a_vec


def clip_polygon_uv(
    subject: np.ndarray,
    clipper: np.ndarray,
    eps_collinear: float,
) -> np.ndarray:
    """
    Clip a convex polygon using the Sutherland-Hodgman algorithm.

    Computes the intersection of two convex polygons in UV space using
    the Sutherland-Hodgman clipping algorithm.

    :param subject: Subject polygon vertices, shape (N, 2).
    :param clipper: Clipping polygon vertices, shape (M, 2).
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Clipped polygon vertices, shape (K, 2) where K is the number of vertices in the result.
    """
    subject = ensure_ccw(subject)
    clipper = ensure_ccw(clipper)

    out = subject
    for i, a_edge in enumerate(clipper):
        input_list = out
        b_edge = clipper[(i + 1) % len(clipper)]

        if len(input_list) == 0:
            return np.zeros((0, 2), dtype=subject.dtype)

        out_list: list[np.ndarray] = []
        for j, p in enumerate(input_list):
            q = input_list[(j + 1) % len(input_list)]

            pin = inside(p, a_edge, b_edge)
            qin = inside(q, a_edge, b_edge)

            if pin and qin:
                out_list.append(q)
            elif pin and not qin:
                out_list.append(intersection_2d(p, q, a_edge, b_edge, eps_collinear))
            elif not pin and qin:
                out_list.append(intersection_2d(p, q, a_edge, b_edge, eps_collinear))
                out_list.append(q)

        if not out_list:
            return np.zeros((0, 2), dtype=subject.dtype)

        out = np.asarray(out_list, dtype=subject.dtype)

    return out


def sort_polygon_by_angle(points: np.ndarray) -> np.ndarray:
    """
    Sort polygon vertices by angle from centroid.

    Orders polygon vertices by their angle relative to the polygon's centroid,
    ensuring a consistent vertex ordering.

    :param points: Array of shape (N, 2) containing polygon vertices.

    :return: Sorted vertices ordered by angle from centroid.
    """
    center = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(ang)]


# pylint: disable=too-many-locals
def compute_clipped_polygon_uv(
    vertices: np.ndarray,
    extent: np.ndarray,
    eps_plane: float = 1e-5,
    eps_collinear: float = 1e-6,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute clipped polygon in UV coordinates and plane frame.

    Computes the intersection of a quad with a 3D bounding box, projecting
    the result to UV coordinates. Returns both the clipped polygon and the
    plane coordinate system for further processing.

    :param vertices: Quad vertices, shape (4, 3) with last vertex as origin.
    :param extent: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
    :param eps_plane: Tolerance for plane distance calculations.
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Tuple of (clipped polygon in UV coordinates, plane frame (P0, u, v, n)).
    """
    p0, u, v, n = plane_from_quad_origin_last(vertices)

    # Extract Z bounds from the vertices and clean the extent
    cleaned_extent = clean_extent_for_intersection(extent, vertices)

    points_box = compute_plane_box_intersections(
        p0, n, cleaned_extent, eps_plane=eps_plane
    )
    if points_box.shape[0] < 2:
        # no meaningful intersection with box
        return np.zeros((0, 2), dtype=np.float64), (p0, u, v, n)

    # project once
    quad_uv = np.asarray(
        [project_to_uv(p, p0, u, v) for p in vertices], dtype=np.float64
    )
    box_uv = np.asarray(
        [project_to_uv(p, p0, u, v) for p in points_box], dtype=np.float64
    )

    quad_uv = sort_polygon_by_angle(quad_uv)
    box_uv = sort_polygon_by_angle(box_uv)

    clipped_uv = clip_polygon_uv(quad_uv, box_uv, eps_collinear=eps_collinear)

    return clipped_uv, (p0, u, v, n)


def compute_pixel_rectangle_from_uv(
    uv: np.ndarray,
    u_cell: float,
    v_cell: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """
    Compute pixel rectangle bounds from UV coordinates.

    Converts UV polygon coordinates to pixel space and computes the bounding
    rectangle that encompasses the polygon. Optimized for rectangular extents.

    :param uv: UV coordinates of polygon vertices, shape (N, 2).
    :param u_cell: Size of one pixel in U direction.
    :param v_cell: Size of one pixel in V direction.
    :param width: Image width in pixels (U direction).
    :param height: Image height in pixels (V direction).

    :return: Tuple of (xmin, ymin, xmax, ymax) pixel coordinates for PIL crop.
    """
    if len(uv) == 0:
        return 0, 0, 0, 0

    # For rectangular extents, directly compute bounds without building full arrays
    u_min, u_max = float(uv[:, 0].min()), float(uv[:, 0].max())
    v_min, v_max = float(uv[:, 1].min()), float(uv[:, 1].max())

    xmin = max(0, int(np.ceil((u_min / u_cell) - 0.5)))
    xmax = min(width, int(np.floor((u_max / u_cell) - 0.5)) + 1)
    ymin = max(0, int(np.ceil((v_min / v_cell) - 0.5)))
    ymax = min(height, int(np.floor((v_max / v_cell) - 0.5)) + 1)

    # Ensure we have a valid rectangle
    if xmin >= xmax or ymin >= ymax:
        return 0, 0, 0, 0

    return xmin, ymin, xmax, ymax


# pylint: disable=too-many-locals
def pixel_rect_to_world(
    extent: tuple[int, int, int, int],
    plane: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    u_cell: float,
    v_cell: float,
) -> np.ndarray:
    """
    Convert pixel rectangle bounds to world coordinates.

    Transforms pixel rectangle corners to 3D world coordinates using the UV
    coordinate system defined by origin p0 and basis vectors u and v.

    The extent contains pixel indices of selected cells. We need the spatial bounds:
    - Left edge of first pixel: xmin * u_cell
    - Right edge of last pixel: (xmax-1 + 1) * u_cell = xmax * u_cell
    - Top edge of first pixel: ymin * v_cell
    - Bottom edge of last pixel: (ymax-1 + 1) * v_cell = ymax * v_cell

    :param extent: Pixel extent as (xmin, ymin, xmax, ymax) where xmax, ymax are exclusive.
    :param plane: Plane coordinate system as (p0, u, v, n).
    :param u_cell: Size of one pixel in U direction.
    :param v_cell: Size of one pixel in V direction.

    :return: Array of shape (4, 3) containing world coordinates of rectangle corners.
    """
    xmin, ymin, xmax, ymax = extent
    p0, u, v, _ = plane

    # Convert pixel indices to spatial coordinates (actual pixel bounds)
    u_min = xmin * u_cell
    u_max = xmax * u_cell
    v_min = ymin * v_cell
    v_max = ymax * v_cell

    # Define corners in UV space
    xs = np.array([u_min, u_max, u_max, u_min], float)
    ys = np.array([v_max, v_max, v_min, v_min], float)

    return np.array([p0 + x * u + y * v for x, y in zip(xs, ys, strict=False)])
