# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

from geoh5py.shared.utils import (
    clean_extent_for_intersection,
    ensure_counter_clockwise,
    normalize,
    validate_3d_array,
    validate_normalized_vector,
)


if TYPE_CHECKING:
    from geoh5py.objects.geo_image import GeoImage
    from geoh5py.objects.grid2d import Grid2D


class Plane(BaseModel):
    """
    Represents a 3D plane with origin and basis vectors.

    :param origin: Origin point of the plane coordinate system.
    :param u_vector: U-axis basis vector.
    :param v_vector: V-axis basis vector.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    origin: Annotated[np.ndarray, AfterValidator(validate_3d_array)]
    u_vector: Annotated[np.ndarray, AfterValidator(validate_normalized_vector)]
    v_vector: Annotated[np.ndarray, AfterValidator(validate_normalized_vector)]

    @model_validator(mode="after")
    def validate_uv(self) -> Plane:
        """
        Ensure u_vector and v_vector are orthogonal.
        Shape and normalization are validated by field validators.
        """
        if not np.isclose(float(np.dot(self.u_vector, self.v_vector)), 0.0, atol=1e-8):
            raise ValueError("u_vector and v_vector are not orthogonal.")

        return self

    @classmethod
    def from_points(
        cls,
        origin: np.ndarray,
        point_u: np.ndarray,
        point_v: np.ndarray,
    ) -> Plane:
        """
        Create a Plane from three points.

        :param origin: Origin point of the plane.
        :param point_u: Point defining the U direction.
        :param point_v: Point defining the V direction.

        :return: Plane object.
        """
        u_vector = normalize(
            np.asarray(point_u, dtype=float) - np.asarray(origin, dtype=float)
        )
        v_vector = normalize(
            np.asarray(point_v, dtype=float) - np.asarray(origin, dtype=float)
        )

        return cls(
            origin=np.asarray(origin, dtype=float), u_vector=u_vector, v_vector=v_vector
        )

    @property
    def dip_rotation_only(self) -> bool:
        """
        Check if the plane orientation can be explained by rotation and dip only.

        :return: True if u_vector lies in the XY plane (has no Z component).
        """
        return abs(self.u_vector[2]) < 1e-6

    @property
    def normal(self) -> np.ndarray:
        """
        Get the normal vector of the plane.

        :return: Normal vector as a numpy array.
        """
        return normalize(np.cross(self.u_vector, self.v_vector))

    def project_to_uv(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        """
        Project a 3D point onto a 2D UV coordinate system.

        Projects a point P onto the UV plane defined by plane object.

        :param point: 3D point to project.

        :return: 2D coordinates in the UV system.
        """
        displacement = np.asarray(point, dtype=np.float64) - self.origin
        return np.array(
            [np.dot(displacement, self.u_vector), np.dot(displacement, self.v_vector)],
            dtype=np.float64,
        )

    def extent_from_vertices_and_box(
        self,
        planar_object: Grid2D | GeoImage,
        vertices: np.ndarray,
        extent: np.ndarray,
        *,
        eps_plane: float = 1e-5,
        eps_collinear: float = 1e-6,
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | tuple[None, None]:
        """
        Compute pixel rectangle extent from quad and bounding box intersection.

        Computes the intersection of a quad with a 3D bounding box, projects
        the result to UV coordinates, and derives the pixel rectangle extent.

        :param planar_object: Object with 'u_count', 'v_count',
            'u_cell_size', 'v_cell_size' attributes.
        :param vertices: Quad vertices, shape (4, 3) with last vertex as origin.
        :param extent: Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
        :param eps_plane: Tolerance for plane distance calculations.
        :param eps_collinear: Tolerance for collinearity detection.

        :return: Pixel rectangle extent as (xmin, ymin, xmax, ymax)
            or None if no valid extent.
        """
        if not isinstance(vertices, np.ndarray) or vertices.shape != (4, 3):
            raise ValueError(
                "vertices must be a numpy array of shape (4, 3). "
                f"Got {type(vertices)} with shape {vertices.shape}."
            )

        if not isinstance(extent, np.ndarray) or extent.shape not in [(2, 3), (2, 2)]:
            raise ValueError(
                "extent must be a numpy array of shape (2, 3)."
                f"Got {type(extent)} with shape {extent.shape}."
            )

        u_count, v_count, u_cell_size, v_cell_size = self._get_count_cell_from_object(
            planar_object
        )

        clipped_uv = self._box_intersection(
            vertices, extent, eps_plane=eps_plane, eps_collinear=eps_collinear
        )

        if clipped_uv.shape[0] < 3:
            return None, None

        new_extent = self._pixel_extent_from_polygon(
            clipped_uv, u_cell_size, v_cell_size, u_count, v_count
        )

        # unlikely if first condition is met
        if new_extent is None:
            return None, None

        new_vertices = self._pixel_rect_to_world(new_extent, u_cell_size, v_cell_size)

        return new_extent, new_vertices

    @staticmethod
    def _clip_polygon_uv(
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

        :return: Clipped polygon vertices,
            shape (K, 2) where K is the number of vertices in the result.
        """
        subject_polygon = ensure_counter_clockwise(subject_polygon)
        clipping_polygon = ensure_counter_clockwise(clipping_polygon)

        output_vertices = subject_polygon
        for i, current_clip_edge_start in enumerate(clipping_polygon):
            input_vertices = output_vertices
            current_clip_edge_end = clipping_polygon[(i + 1) % len(clipping_polygon)]

            if len(input_vertices) == 0:
                return np.empty((0, 2), dtype=subject_polygon.dtype)

            clipped_vertices: list[np.ndarray] = []
            for j, current_vertex in enumerate(input_vertices):
                next_vertex = input_vertices[(j + 1) % len(input_vertices)]

                current_inside = Plane._point_left_of_directed_edge(
                    current_vertex, current_clip_edge_start, current_clip_edge_end
                )
                next_inside = Plane._point_left_of_directed_edge(
                    next_vertex, current_clip_edge_start, current_clip_edge_end
                )

                if current_inside and next_inside:
                    clipped_vertices.append(next_vertex)
                elif current_inside and not next_inside:
                    clipped_vertices.append(
                        Plane._intersection_2d(
                            current_vertex,
                            next_vertex,
                            current_clip_edge_start,
                            current_clip_edge_end,
                            eps_collinear,
                        )
                    )
                elif not current_inside and next_inside:
                    clipped_vertices.append(
                        Plane._intersection_2d(
                            current_vertex,
                            next_vertex,
                            current_clip_edge_start,
                            current_clip_edge_end,
                            eps_collinear,
                        )
                    )
                    clipped_vertices.append(next_vertex)

            if not clipped_vertices:
                return np.empty((0, 2), dtype=subject_polygon.dtype)

            output_vertices = np.asarray(clipped_vertices, dtype=subject_polygon.dtype)

        return output_vertices

    @staticmethod
    def _intersection_2d(
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
            offset_vector[0] * segment2_vector[1]
            - offset_vector[1] * segment2_vector[0]
        ) / denominator
        return segment1_start + parameter_t * segment1_vector

    @staticmethod
    def _point_left_of_directed_edge(
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

    @staticmethod
    def _sort_polygon_by_angle(polygon_vertices: np.ndarray) -> np.ndarray:
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

    def _intersect_segment_plane(
        self,
        start_point: np.ndarray,
        end_point: np.ndarray,
        *,
        eps_plane: float = 1e-5,
    ) -> np.ndarray | None:
        """
        Compute intersection of a line segment with a plane.

        Finds the intersection point between a line segment defined by points A and B
        and a plane object.

        :param start_point: Start point of the line segment.
        :param end_point: End point of the line segment.
        :param eps_plane: Tolerance for plane distance calculations.

        :return: Intersection point if it exists within the segment, None otherwise.
        """
        distance_start = np.dot(start_point - self.origin, self.normal)
        distance_end = np.dot(end_point - self.origin, self.normal)

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

    def _compute_box_intersections(
        self,
        extent: np.ndarray,
        eps_plane: float,
    ) -> np.ndarray:
        """
        Compute intersections between a plane and the edges of a 3D bounding box.

        Finds all points where the plane intersects with the 12 edges of a 3D box
        defined by its min/max coordinates.

        :param extent: coordinates of (xmin, xmax, ymin, ymax, zmin, zmax) defining the box.
        :param eps_plane: Tolerance for plane distance calculations.

        :return: Array of intersection points, shape (N, 3) where N is the number of intersections.
        """
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
            intersection = self._intersect_segment_plane(
                box_corners[corner_a], box_corners[corner_b], eps_plane=eps_plane
            )
            if intersection is not None:
                intersection_points.append(intersection)

        if not intersection_points:
            return np.zeros((0, 3), dtype=np.float64)

        return np.asarray(intersection_points, dtype=np.float64)

    def _pixel_rect_to_world(
        self,
        pixel_extent: tuple[int, int, int, int],
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

        :param pixel_extent: Pixel extent as (xmin, ymin, xmax, ymax)
            where xmax, ymax are exclusive.
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
                self.origin + u_coord * self.u_vector + v_coord * self.v_vector
                for u_coord, v_coord in zip(u_coordinates, v_coordinates, strict=False)
            ]
        )

    @staticmethod
    def _pixel_extent_from_polygon(
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

    def _box_intersection(
        self,
        vertices: np.ndarray,
        extent: np.ndarray,
        eps_plane: float = 1e-5,
        eps_collinear: float = 1e-6,
    ) -> np.ndarray:
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
        # Extract Z bounds from the vertices and clean the extent
        cleaned_extent = clean_extent_for_intersection(extent, vertices)

        box_intersection_points = self._compute_box_intersections(
            cleaned_extent, eps_plane=eps_plane
        )
        if box_intersection_points.shape[0] < 2:
            # no meaningful intersection with box
            return np.zeros((0, 2), dtype=np.float64)

        # project once
        quad_uv_coordinates = np.asarray(
            [self.project_to_uv(vertex) for vertex in vertices], dtype=np.float64
        )
        box_uv_coordinates = np.asarray(
            [self.project_to_uv(point) for point in box_intersection_points],
            dtype=np.float64,
        )

        quad_uv_sorted = self._sort_polygon_by_angle(quad_uv_coordinates)
        box_uv_sorted = self._sort_polygon_by_angle(box_uv_coordinates)

        clipped_uv_polygon = self._clip_polygon_uv(
            quad_uv_sorted, box_uv_sorted, eps_collinear=eps_collinear
        )

        return clipped_uv_polygon

    @staticmethod
    def _get_count_cell_from_object(
        planar_object: Grid2D | GeoImage,
    ) -> tuple[int, int, int, int]:
        """
        Get U and V pixel counts from a planar object.

        :param planar_object: Object with 'u_count', 'v_count',
            'u_cell_size', 'v_cell_size' attributes.

        :return: Tuple of (u_count, v_count, u_cell_size, v_cell_size).
        """
        attributes = ["u_count", "v_count", "u_cell_size", "v_cell_size"]

        output = []
        for attribute in attributes:
            if not hasattr(planar_object, attribute):
                raise ValueError(f"Planar object must have '{attribute}' attribute.")
            output.append(getattr(planar_object, attribute))

        return output[0], output[1], output[2], output[3]
