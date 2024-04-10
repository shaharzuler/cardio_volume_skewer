from typing import Tuple
import numpy as np

import flow_n_corr_utils

class ErrorAnalysisCoordinateSystem:
    def __init__(self, required_shape: Tuple, x, y, z, center, orig_shell_vertices_mean, main_rotation_matrix, flow_rotator) -> None:
        self.shape = required_shape
        self.x = x
        self.y = y
        self.z = z
        self.center = center
        self.orig_shell_vertices_mean = orig_shell_vertices_mean
        self.main_rotation_matrix = main_rotation_matrix
        self.flow_rotator = flow_rotator

        self.radial_coordinate_base = self._compute_radial_coordinate_base()
        self.longitudinal_coordinate_base = self._compute_longitudinal_coordinate_base()
        self.circumferential_coordinate_base = self._compute_circumferential_coordinate_base()

        self._adjust_all_coordinates()

    def _compute_radial_coordinate_base(self):
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        center = np.array([self.shape[1]//2, self.shape[0]//2])  

        xy_vector_components = np.stack([center[0] - x, center[1] - y], axis=2)
        xy_vector_magnitudes = np.linalg.norm(xy_vector_components, axis=2, keepdims=True)
        xy_vectors_normed = np.nan_to_num((xy_vector_components / xy_vector_magnitudes), nan=0.0)

        xyz_vectors = np.concatenate([xy_vectors_normed, np.zeros( (self.shape[0], self.shape[1], 1), dtype=xy_vectors_normed.dtype)], axis=2)
        radial_coordinate_base=np.repeat(np.expand_dims(xyz_vectors,2), self.shape[2], axis=2)

        return radial_coordinate_base

    def _compute_longitudinal_coordinate_base(self):
        longitudinal_coordinate_base = np.zeros((*self.shape, 3), dtype=float)
        longitudinal_coordinate_base[:,:,:,2] = 1.
        return longitudinal_coordinate_base

    def _compute_circumferential_coordinate_base(self):
        circumferential_coordinate_base = np.cross(self.radial_coordinate_base, self.longitudinal_coordinate_base, axis=3)
        return circumferential_coordinate_base
   
    def _adjust_single_coordinates_axis(self, axis_coordinate_base):
        axis_coordinate_base_rotated = self.flow_rotator.rotate_flow(axis_coordinate_base, np.linalg.inv(self.main_rotation_matrix))
        axis_coordinate_base_cropped = flow_n_corr_utils.crop_flow_by_mask_center(self.center, self.x, self.y, self.z, axis_coordinate_base_rotated, self.orig_shell_vertices_mean)
        axis_coordinate_base_filled = flow_n_corr_utils.interp_to_fill_nans(axis_coordinate_base_cropped)
        return axis_coordinate_base_filled

    def _adjust_all_coordinates(self):
        self.radial_coordinate_base = self._adjust_single_coordinates_axis(self.radial_coordinate_base)
        self.circumferential_coordinate_base = self._adjust_single_coordinates_axis(self.circumferential_coordinate_base)
        self.longitudinal_coordinate_base = self._adjust_single_coordinates_axis(self.longitudinal_coordinate_base)