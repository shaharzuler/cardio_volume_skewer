from typing import Tuple
import math
import os

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial.transform import Rotation as R
from scipy.spatial.qhull import QhullError
import nrrd
import torch
from torch import nn
import patchify
from matplotlib import pyplot as plt

import three_d_data_manager
import flow_n_corr_utils

from .error_analysis_tools.error_analysis_coordinate_system import ErrorAnalysisCoordinateSystem


class VolumeSkewer:
    def __init__(self, save_nrrd:bool=True, save_npy:bool=True, zero_outside_mask:bool=True, blur_around_mask_radious:int=0, warping_borders_pad='zeros', image_warping_interp_mode='bilinear', mask_warping_interp_mode='nearest', theta_changing_method='linear'):
        self.save_nrrd = save_nrrd
        self.save_npy = save_npy
        self.zero_outside_mask = zero_outside_mask
        self.blur_around_mask_radious = blur_around_mask_radious
        self.warping_borders_pad=warping_borders_pad
        self.image_warping_interp_mode = image_warping_interp_mode
        self.mask_warping_interp_mode = mask_warping_interp_mode
        self.theta_changing_method=theta_changing_method
        
    def skew_volume(
        self,  theta1:float, theta2:float, r1:float, r2:float, h:float, \
            three_d_image:np.array, three_d_binary_mask:np.array, extra_three_d_binary_mask:np.array, output_dir:str) -> None: 

        self.three_d_image = three_d_image
        self.three_d_binary_mask = three_d_binary_mask
        self.extra_three_d_binary_mask = extra_three_d_binary_mask
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.theta1 = theta1
        self.theta2 = theta2        
        self.r1 = r1
        self.r2 = r2
        self.h = h
        self.x, self.y, self.z = three_d_image.shape
        self.center = (np.array([three_d_image.shape])-1) / 2
        self.main_axis = np.array([0, 0, 1])
        self.main_rotation_matrix = self._compute_main_rotation()
        
        self.x_flow = 2 * self.x
        self.y_flow = 2 * self.y
        self.z_flow = 2 * self.z
        self.center_flow = 2 * self.center
        
        self.total_torsion = self._compute_total_torsion()
        
        flow_field = self._init_flow_field()
        rs = np.linspace(r1, r2, self.z_flow)
        thetas = self._get_thetas(theta1, theta2)
        flow_field = self._create_flow_for_r_and_scale(flow_field, rs, thetas)
        self.flow_field = self._stretch_flow_vertically(flow_field)

        self.flow_rotator = flow_n_corr_utils.FlowRotator(self.x_flow, self.y_flow, self.z_flow, self.center_flow)
        self.flow_field_rotated = self.flow_rotator.rotate_flow(flow_field, np.linalg.inv(self.main_rotation_matrix))
        self.flow_for_mask = flow_n_corr_utils.crop_flow_by_mask_center(self.center, self.x, self.y, self.z, self.flow_field_rotated, self.orig_shell_vertices_mean)
        self.flow_for_mask = flow_n_corr_utils.interp_to_fill_nans(self.flow_for_mask)
        self.error_analysis_coordinate_system = self._get_error_analysis_coordinate_system((self.x_flow, self.y_flow, self.z_flow))
    
    def handle_outside_mask(self):
        mask = self.skewed_three_d_binary_mask.astype(bool)
        
        if self.blur_around_mask_radious > 0:
            dilated_mask_layers = []
            largets_mask = mask.copy()
            for r in range(self.blur_around_mask_radious):
                dilated_mask = ndimage.binary_dilation(largets_mask)
                dilated_mask_layers.append(dilated_mask^largets_mask)
                largets_mask = dilated_mask.copy()
            out_mask = ~largets_mask
            for n, layer in enumerate(dilated_mask_layers):
                self.scaled_flow_for_mask[layer.nonzero()] *= 1. - (float(n+1)/self.blur_around_mask_radious)
        else:
            out_mask = ~mask
        self.scaled_flow_for_mask[out_mask.nonzero()] = 0.

        return self.scaled_flow_for_mask

    def save_nrrds(self, suffix:str)->None:
        # nrrd.write(os.path.join(self.output_dir, f'image_orig{suffix}.nrrd'),        self.three_d_image)
        nrrd.write(os.path.join(self.output_dir, f'image_skewed{suffix}.nrrd'),      self.skewed_three_d_image)
        # nrrd.write(os.path.join(self.output_dir, f'mask_orig{suffix}.nrrd'),         self.three_d_binary_mask.astype(float))
        nrrd.write(os.path.join(self.output_dir, f'mask_skewed{suffix}.nrrd'),       self.skewed_three_d_binary_mask.astype(float))
        nrrd.write(os.path.join(self.output_dir, f'extra_mask_skewed{suffix}.nrrd'), self.skewed_extra_three_d_binary_mask.astype(float))

    def save_npys(self, suffix:str)->None:
        np.save(os.path.join(self.output_dir, f'image_orig{suffix}.npy'),        self.three_d_image)
        np.save(os.path.join(self.output_dir, f'image_skewed{suffix}.npy'),      self.skewed_three_d_image)
        # np.save(os.path.join(self.output_dir, f"flow_for_mask{suffix}.npy"),     self.scaled_flow_for_mask)
        np.save(os.path.join(self.output_dir, f"flow_for_image{suffix}.npy"),    self.scaled_flow_for_image)
        np.save(os.path.join(self.output_dir, f'mask_orig{suffix}.npy'),         self.three_d_binary_mask.astype(bool))
        np.save(os.path.join(self.output_dir, f'extra_mask_orig{suffix}.npy'),   self.extra_three_d_binary_mask.astype(bool))
        np.save(os.path.join(self.output_dir, f'mask_skewed{suffix}.npy'),       self.skewed_three_d_binary_mask.astype(bool))
        np.save(os.path.join(self.output_dir, f'extra_mask_skewed{suffix}.npy'), self.skewed_extra_three_d_binary_mask.astype(bool))

        np.save(os.path.join(self.output_dir, f'error_radial_coordinates{suffix}.npy'), self.error_analysis_coordinate_system.radial_coordinate_base)
        np.save(os.path.join(self.output_dir, f'error_circumferential_coordinates{suffix}.npy'), self.error_analysis_coordinate_system.circumferential_coordinate_base)
        np.save(os.path.join(self.output_dir, f'error_longitudinal_coordinates{suffix}.npy'), self.error_analysis_coordinate_system.longitudinal_coordinate_base)

    def _compute_main_rotation(self)->np.array:
        shell = three_d_data_manager.extract_segmentation_envelope(self.three_d_binary_mask)
        self.orig_shell_vertices =np.array(shell.nonzero()).T

        self.orig_shell_vertices_mean = self.orig_shell_vertices.mean(axis=0)
        self.orig_shell_vertices_centered = self.orig_shell_vertices - self.orig_shell_vertices_mean
        
        U, S, Vt = np.linalg.svd(self.orig_shell_vertices_centered)

        x_to_z = np.array([[0,0,-1],[0,-1,0],[1,0,0]])
        return (x_to_z @ Vt).T

    def _compute_mask_pixels_range_over_principal_axis(self):
        mask_projected_coords_over_principal_axis = np.dot(self.orig_shell_vertices_centered, self.main_rotation_matrix)[:,-1]
        return round(mask_projected_coords_over_principal_axis.max() - mask_projected_coords_over_principal_axis.min())
        
    def _compute_total_torsion(self):
        self.mask_pixels_range_over_principal_axis = self._compute_mask_pixels_range_over_principal_axis()
        total_torsion = (self.theta2 - self.theta1) * (float(self.mask_pixels_range_over_principal_axis) / self.z_flow)
        print(f"Total torsion: {total_torsion}")
        with open(os.path.join(self.output_dir, 'total_torsion.txt'), 'w') as f:
            f.write(f"Total torsion: {total_torsion}")
        return total_torsion

    def _init_flow_field(self)->np.array:
        flow_field = np.empty([self.x_flow, self.y_flow, self.z_flow, 3])  
        flow_field[:] = np.nan
        return flow_field

    def _create_flow_for_r_and_scale(self, flow_field:np.array, rs:np.array, thetas:np.array)->np.array:
        for xy_plane_i in range(self.z_flow):
            edges_coords = np.array(
                (
                [0,             0,             xy_plane_i],
                [0,             self.y_flow-1, xy_plane_i], 
                [self.x_flow-1, 0,             xy_plane_i],
                [self.x_flow-1, self.y_flow-1, xy_plane_i] 
                )
            )
            edges_coords_centered = edges_coords - self.center_flow # z axis doesn't need to be reduced but it doesnt matter since the rotation is around z axis
            r = R.from_rotvec((thetas[xy_plane_i] * (math.pi / 180)) * self.main_axis)
            rot_matrix = r.as_matrix()
            scale_matrix = np.array(
                [
                    [rs[xy_plane_i], 0,                   0], 
                    [0,                   rs[xy_plane_i], 0],
                    [0,                   0,              1]
                ]
            )
            transform_matrix = scale_matrix @ rot_matrix 
            edges_coords_moved = (edges_coords_centered @ transform_matrix) + self.center_flow
            flow_field[edges_coords[:,0], edges_coords[:,1], edges_coords[:,2]] = edges_coords_moved - edges_coords

            flow_field[:, :, xy_plane_i, 0] = flow_n_corr_utils.interp_missing_values(flow_field[:, :, xy_plane_i, 0].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 1] = flow_n_corr_utils.interp_missing_values(flow_field[:, :, xy_plane_i, 1].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 2] = flow_n_corr_utils.interp_missing_values(flow_field[:, :, xy_plane_i, 2].copy(), LinearNDInterpolator)
        return flow_field
    
    def _stretch_flow_vertically(self, flow_field:np.array)->np.array:
        z0_to_center = np.linspace((-(self.z_flow/2) * self.h) + (self.z_flow/2), 0, self.z_flow//2)
        flow_field[:, :, :self.z_flow//2, 2] = z0_to_center
        flow_field[:, :, self.z_flow//2:, 2] = -z0_to_center[::-1]
        return flow_field

    def _get_error_analysis_coordinate_system(self, required_shape): 
        error_analysis_coordinate_system = ErrorAnalysisCoordinateSystem(
            required_shape, 
            self.x, self.y, self.z, 
            self.center, self.orig_shell_vertices_mean,
            self.main_rotation_matrix, self.flow_rotator
            )
        return error_analysis_coordinate_system

    def _get_thetas(self, theta1:float, theta2:float) -> np.ndarray:
        epsilon = 0.1
        if self.theta_changing_method == "linear":
            thetas = np.linspace(theta1, theta2, self.z_flow)
        elif self.theta_changing_method == "geometric":
            max_ = abs(theta2-theta1)
            thetas = np.geomspace(epsilon, max_, self.z_flow) 
            if theta1 > theta2:
                thetas = thetas[::-1]
            thetas = thetas + min(abs(theta1), abs(theta2)) - epsilon
        elif "random" in self.theta_changing_method:
            noise_mag = float(self.theta_changing_method.split("_")[-2])
            noise_bias = float(self.theta_changing_method.split("_")[-1])
            progress = (np.random.rand(self.z_flow) - noise_bias) * noise_mag
            thetas = np.cumsum(progress)

        plt.plot(thetas)
        plt.title("thetas")
        plt.savefig(os.path.join(self.output_dir, "thetas.jpg"))
        return thetas

