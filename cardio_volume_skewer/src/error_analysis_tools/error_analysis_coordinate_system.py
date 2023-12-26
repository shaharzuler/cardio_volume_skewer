from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

class ErrorAnalysisCoordinateSystem:
    def __init__(self, required_shape: Tuple) -> None:
        self.shape = required_shape
        self.radial_coordinate_base = self.compute_radial_coordinate_base()
        self.longitudinal_coordinate_base = self.compute_longitudinal_coordinate_base()
        self.circumferential_coordinate_base = self.compute_circumferential_coordinate_base()

    def compute_radial_coordinate_base(self):
        x, y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        center = np.array([self.shape[1]//2, self.shape[0]//2])  

        xy_vector_components = np.stack([center[0] - x, center[1] - y], axis=2)
        xy_vector_magnitudes = np.linalg.norm(xy_vector_components, axis=2, keepdims=True)
        xy_vectors_normed = np.nan_to_num((xy_vector_components / xy_vector_magnitudes), nan=0.0)

        xyz_vectors = np.concatenate([xy_vectors_normed, np.zeros( (self.shape[0], self.shape[1], 1), dtype=xy_vectors_normed.dtype)], axis=2)
        radial_coordinate_base=np.repeat(np.expand_dims(xyz_vectors,2), self.shape[2], axis=2)

        return radial_coordinate_base

    def compute_longitudinal_coordinate_base(self):
        longitudinal_coordinate_base = np.zeros((*self.shape, 3), dtype=float)
        longitudinal_coordinate_base[:,:,:,2] = 1.
        return longitudinal_coordinate_base

    def compute_circumferential_coordinate_base(self):
        circumferential_coordinate_base = np.cross(self.radial_coordinate_base, self.longitudinal_coordinate_base, axis=3)
        return circumferential_coordinate_base

        



