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

from dataclasses import dataclass

import numpy as np

from geoh5py.shared.utils import clean_extent_for_intersection


@dataclass
class Plane:
    """
    Represents a 3D plane with origin and basis vectors.

    :param origin: Origin point of the plane coordinate system.
    :param u_vector: U-axis basis vector.
    :param v_vector: V-axis basis vector.
    :param normal: Normal vector of the plane.
    """

    origin: np.ndarray
    u_vector: np.ndarray
    v_vector: np.ndarray
    normal: np.ndarray

    @classmethod
    def from_quad_origin_last(cls, vertices: np.ndarray) -> Plane:
        """
        Create a coordinate system from a quad with the last vertex as origin.

        Constructs a plane coordinate system (origin, u-vector, v-vector, normal)
        from four vertices where the last vertex serves as the origin.

        :param vertices: Array of shape (4, 3) containing the four vertices of the quad.

        :return: Plane object containing origin and basis vectors.

        :raises ValueError: If vertices is not a (4, 3) array.
        """
        if vertices.shape != (4, 3):
            raise ValueError("vertices must be a (4, 3) array")

        vertices = np.asarray(vertices)

        origin = vertices[3]
        u_vector = normalize(vertices[2] - origin)
        v_vector = normalize(vertices[0] - origin)
        normal = normalize(np.cross(u_vector, v_vector))
        return cls(origin=origin, u_vector=u_vector, v_vector=v_vector, normal=normal)


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    :param vector: Input vector to normalize.
    :param dtype: NumPy data type for the normalized vector.

    :return: Normalized vector with unit length.
    """
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def intersect_segment_plane(
    start_point: np.ndarray,
    end_point: np.ndarray,
    plane: Plane,
    *,
    eps_plane: float = 1e-5,
) -> np.ndarray | None:
    """
    Compute intersection of a line segment with a plane.

    Finds the intersection point between a line segment defined by points A and B
    and a plane object.

    :param start_point: Start point of the line segment.
    :param end_point: End point of the line segment.
    :param plane: Plane object containing origin and normal vector.
    :param eps_plane: Tolerance for plane distance calculations.

    :return: Intersection point if it exists within the segment, None otherwise.
    """
    distance_start = np.dot(start_point - plane.origin, plane.normal)
    distance_end = np.dot(end_point - plane.origin, plane.normal)

    # Both on same side or both almost on plane = treat as no crossing
    if distance_start * distance_end > 0 or (
        abs(distance_start) < eps_plane and abs(distance_end) < eps_plane
    ):
        return None

    denominator = distance_start - distance_end
    if abs(denominator) < eps_plane:
        return None

    parameter_t = distance_start / denominator
    if 0.0 <= parameter_t <= 1.0:
        return start_point + parameter_t * (end_point - start_point)
    return None


def compute_plane_box_intersections(
    plane: Plane,
    extent: np.ndarray,
    eps_plane: float,
) -> np.ndarray:
    """
    Compute intersections between a plane and the edges of a 3D bounding box.

    Finds all points where the plane intersects with the 12 edges of a 3D box
    defined by its min/max coordinates.

    :param plane: Plane object containing origin and normal vector.
    :param extent: coordinates of (xmin, xmax, ymin, ymax, zmin, zmax) defining the box.
    :param eps_plane: Tolerance for plane distance calculations.

    :return: Array of intersection points, shape (N, 3) where N is the number of intersections.
    """
    # Expect extent to be already cleaned (shape (2, 3))
    x_min, y_min, z_min = extent[0]
    x_max, y_max, z_max = extent[1]

    box_corners = np.asarray(
        [
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ],
        dtype=np.float64,
    )

    box_edges = [
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

    intersection_points: list[np.ndarray] = []
    for corner_a, corner_b in box_edges:
        intersection = intersect_segment_plane(
            box_corners[corner_a], box_corners[corner_b], plane, eps_plane=eps_plane
        )
        if intersection is not None:
            intersection_points.append(intersection)

    if not intersection_points:
        return np.zeros((0, 3), dtype=np.float64)

    return np.asarray(intersection_points, dtype=np.float64)


def project_to_uv(
    point: np.ndarray,
    plane: Plane,
) -> np.ndarray:
    """
    Project a 3D point onto a 2D UV coordinate system.

    Projects a point P onto the UV plane defined by plane object.

    :param point: 3D point to project.
    :param plane: Plane object containing origin and basis vectors.

    :return: 2D coordinates in the UV system.
    """
    displacement = np.asarray(point, dtype=np.float64) - plane.origin
    return np.array(
        [np.dot(displacement, plane.u_vector), np.dot(displacement, plane.v_vector)],
        dtype=np.float64,
    )


def ensure_ccw(polygon: np.ndarray) -> np.ndarray:
    """
    Ensure polygon vertices are ordered counter-clockwise.

    Reverses the vertex order if the polygon area is negative (clockwise orientation).

    :param polygon: Array of shape (N, 2) containing polygon vertices.

    :return: Polygon vertices in counter-clockwise order.
    """
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    polygon_area = 0.5 * float(
        np.sum(x_coords * np.roll(y_coords, -1) - y_coords * np.roll(x_coords, -1))
    )

    if polygon_area < 0.0:
        return polygon[::-1]

    return polygon


def point_left_of_directed_edge(
    test_point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray
) -> bool:
    """
    Test if a point is on the left side of a directed edge using 2D cross product.

    Computes the cross product of the edge vector and the vector from edge start
    to the test point. A positive result means the point is on the left side
    of the directed edge (counter-clockwise orientation).

    Used in polygon clipping to determine if a point is on the "inside"
    of a clipping edge when the polygon is oriented counter-clockwise.

    :param test_point: Point to test.
    :param edge_start: Start point of the edge.
    :param edge_end: End point of the edge.

    :return: True if the point is inside (left of) the edge, False otherwise.
    """
    return (edge_end[0] - edge_start[0]) * (test_point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (test_point[0] - edge_start[0]) >= 0.0


def intersection_2d(
    segment1_start: np.ndarray,
    segment1_end: np.ndarray,
    segment2_start: np.ndarray,
    segment2_end: np.ndarray,
    eps_collinear: float,
) -> np.ndarray:
    """
    Compute intersection point of two 2D line segments.

    Finds the intersection point of line segment p1-p2 with line segment a-b.
    If the segments are collinear, returns p2 as a fallback.

    :param segment1_start: Start point of the first segment.
    :param segment1_end: End point of the first segment.
    :param segment2_start: Start point of the second segment.
    :param segment2_end: End point of the second segment.
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Intersection point or segment1_end if segments are collinear.
    """
    segment1_vector = segment1_end - segment1_start
    segment2_vector = segment2_end - segment2_start
    offset_vector = segment2_start - segment1_start
    denominator = (
        segment1_vector[0] * segment2_vector[1]
        - segment1_vector[1] * segment2_vector[0]
    )
    if abs(denominator) < eps_collinear:
        return segment1_end
    parameter_t = (
        offset_vector[0] * segment2_vector[1] - offset_vector[1] * segment2_vector[0]
    ) / denominator
    return segment1_start + parameter_t * segment1_vector


def clip_polygon_uv(
    subject_polygon: np.ndarray,
    clipping_polygon: np.ndarray,
    eps_collinear: float,
) -> np.ndarray:
    """
    Clip a convex polygon using the Sutherland-Hodgman algorithm.

    Computes the intersection of two convex polygons in UV space using
    the Sutherland-Hodgman clipping algorithm.

    :param subject_polygon: Subject polygon vertices, shape (N, 2).
    :param clipping_polygon: Clipping polygon vertices, shape (M, 2).
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Clipped polygon vertices, shape (K, 2) where K is the number of vertices in the result.
    """
    subject_polygon = ensure_ccw(subject_polygon)
    clipping_polygon = ensure_ccw(clipping_polygon)

    output_vertices = subject_polygon
    for i, current_clip_edge_start in enumerate(clipping_polygon):
        input_vertices = output_vertices
        current_clip_edge_end = clipping_polygon[(i + 1) % len(clipping_polygon)]

        if len(input_vertices) == 0:
            return np.zeros((0, 2), dtype=subject_polygon.dtype)

        clipped_vertices: list[np.ndarray] = []
        for j, current_vertex in enumerate(input_vertices):
            next_vertex = input_vertices[(j + 1) % len(input_vertices)]

            current_inside = point_left_of_directed_edge(
                current_vertex, current_clip_edge_start, current_clip_edge_end
            )
            next_inside = point_left_of_directed_edge(
                next_vertex, current_clip_edge_start, current_clip_edge_end
            )

            if current_inside and next_inside:
                clipped_vertices.append(next_vertex)
            elif current_inside and not next_inside:
                clipped_vertices.append(
                    intersection_2d(
                        current_vertex,
                        next_vertex,
                        current_clip_edge_start,
                        current_clip_edge_end,
                        eps_collinear,
                    )
                )
            elif not current_inside and next_inside:
                clipped_vertices.append(
                    intersection_2d(
                        current_vertex,
                        next_vertex,
                        current_clip_edge_start,
                        current_clip_edge_end,
                        eps_collinear,
                    )
                )
                clipped_vertices.append(next_vertex)

        if not clipped_vertices:
            return np.zeros((0, 2), dtype=subject_polygon.dtype)

        output_vertices = np.asarray(clipped_vertices, dtype=subject_polygon.dtype)

    return output_vertices


def sort_polygon_by_angle(polygon_vertices: np.ndarray) -> np.ndarray:
    """
    Sort polygon vertices by angle from centroid.

    Orders polygon vertices by their angle relative to the polygon's centroid,
    ensuring a consistent vertex ordering.

    :param polygon_vertices: Array of shape (N, 2) containing polygon vertices.

    :return: Sorted vertices ordered by angle from centroid.
    """
    centroid = polygon_vertices.mean(axis=0)
    angles = np.arctan2(
        polygon_vertices[:, 1] - centroid[1], polygon_vertices[:, 0] - centroid[0]
    )
    return polygon_vertices[np.argsort(angles)]


def plane_box_intersection(
    vertices: np.ndarray,
    extent: np.ndarray,
    eps_plane: float = 1e-5,
    eps_collinear: float = 1e-6,
) -> tuple[np.ndarray, Plane]:
    """
    Compute clipped polygon in UV coordinates and plane frame.

    Computes the intersection of a quad with a 3D bounding box, projecting
    the result to UV coordinates. Returns both the clipped polygon and the
    plane coordinate system for further processing.

    :param vertices: Quad vertices, shape (4, 3) with last vertex as origin.
    :param extent: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
    :param eps_plane: Tolerance for plane distance calculations.
    :param eps_collinear: Tolerance for collinearity detection.

    :return: Tuple of (clipped polygon in UV coordinates, plane object).
    """
    plane = Plane.from_quad_origin_last(vertices)

    # Extract Z bounds from the vertices and clean the extent
    cleaned_extent = clean_extent_for_intersection(extent, vertices)

    box_intersection_points = compute_plane_box_intersections(
        plane, cleaned_extent, eps_plane=eps_plane
    )
    if box_intersection_points.shape[0] < 2:
        # no meaningful intersection with box
        return np.zeros((0, 2), dtype=np.float64), plane

    # project once
    quad_uv_coordinates = np.asarray(
        [project_to_uv(vertex, plane) for vertex in vertices], dtype=np.float64
    )
    box_uv_coordinates = np.asarray(
        [project_to_uv(point, plane) for point in box_intersection_points],
        dtype=np.float64,
    )

    quad_uv_sorted = sort_polygon_by_angle(quad_uv_coordinates)
    box_uv_sorted = sort_polygon_by_angle(box_uv_coordinates)

    clipped_uv_polygon = clip_polygon_uv(
        quad_uv_sorted, box_uv_sorted, eps_collinear=eps_collinear
    )

    return clipped_uv_polygon, plane


def pixel_extent_from_polygon(
    uv_coordinates: np.ndarray,
    u_cell_size: float,
    v_cell_size: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Compute pixel rectangle bounds from UV coordinates.

    Converts UV polygon coordinates to pixel space and computes the bounding
    rectangle that encompasses the polygon. Optimized for rectangular extents.

    :param uv_coordinates: UV coordinates of polygon vertices, shape (N, 2).
    :param u_cell_size: Size of one pixel in U direction.
    :param v_cell_size: Size of one pixel in V direction.
    :param image_width: Image width in pixels (U direction).
    :param image_height: Image height in pixels (V direction).

    :return: Tuple of (xmin, ymin, xmax, ymax) pixel coordinates for PIL crop.
        None if no valid rectangle can be formed.
    """
    if len(uv_coordinates) == 0:
        return None

    # For rectangular extents, directly compute bounds without building full arrays
    u_minimum, u_maximum = (
        float(uv_coordinates[:, 0].min()),
        float(uv_coordinates[:, 0].max()),
    )
    v_minimum, v_maximum = (
        float(uv_coordinates[:, 1].min()),
        float(uv_coordinates[:, 1].max()),
    )

    x_min = max(0, int(np.ceil((u_minimum / u_cell_size) - 0.5)))
    x_max = min(image_width, int(np.floor((u_maximum / u_cell_size) - 0.5)) + 1)
    y_min = max(0, int(np.ceil((v_minimum / v_cell_size) - 0.5)))
    y_max = min(image_height, int(np.floor((v_maximum / v_cell_size) - 0.5)) + 1)

    # Ensure we have a valid rectangle
    if x_min >= x_max or y_min >= y_max:
        return None

    return x_min, y_min, x_max, y_max


def pixel_rect_to_world(
    pixel_extent: tuple[int, int, int, int],
    plane: Plane,
    u_cell_size: float,
    v_cell_size: float,
) -> np.ndarray:
    """
    Convert pixel rectangle bounds to world coordinates.

    Transforms pixel rectangle corners to 3D world coordinates using the UV
    coordinate system defined by the plane object.

    The extent contains pixel indices of selected cells. We need the spatial bounds:
    - Left edge of first pixel: xmin * u_cell
    - Right edge of last pixel: (xmax-1 + 1) * u_cell = xmax * u_cell
    - Top edge of first pixel: ymin * v_cell
    - Bottom edge of last pixel: (ymax-1 + 1) * v_cell = ymax * v_cell

    :param pixel_extent: Pixel extent as (xmin, ymin, xmax, ymax) where xmax, ymax are exclusive.
    :param plane: Plane object containing origin and basis vectors.
    :param u_cell_size: Size of one pixel in U direction.
    :param v_cell_size: Size of one pixel in V direction.

    :return: Array of shape (4, 3) containing world coordinates of rectangle corners.
    """
    x_min, y_min, x_max, y_max = pixel_extent

    # Convert pixel indices to spatial coordinates (actual pixel bounds)
    u_minimum = x_min * u_cell_size
    u_maximum = x_max * u_cell_size
    v_minimum = y_min * v_cell_size
    v_maximum = y_max * v_cell_size

    # Define corners in UV space
    u_coordinates = np.array([u_minimum, u_maximum, u_maximum, u_minimum], float)
    v_coordinates = np.array([v_maximum, v_maximum, v_minimum, v_minimum], float)

    return np.array(
        [
            plane.origin + u_coord * plane.u_vector + v_coord * plane.v_vector
            for u_coord, v_coord in zip(u_coordinates, v_coordinates, strict=False)
        ]
    )
