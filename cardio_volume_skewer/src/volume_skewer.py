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

import three_d_data_manager



class VolumeSkewer:
    def __init__(self, save_nrrd:bool=True, save_npy:bool=True, zero_outside_mask:bool=True, blur_around_mask_radious:int=0, warping_borders_pad='zeros', image_warping_interp_mode='bilinear', mask_warping_interp_mode='nearest'):
        self.save_nrrd = save_nrrd
        self.save_npy = save_npy
        self.zero_outside_mask = zero_outside_mask
        self.blur_around_mask_radious = blur_around_mask_radious
        self.warping_borders_pad=warping_borders_pad
        self.image_warping_interp_mode = image_warping_interp_mode
        self.mask_warping_interp_mode = mask_warping_interp_mode
        
    def skew_volume(self,  theta1:float, theta2:float, r1:float, r2:float, h:float, three_d_image:np.array, three_d_binary_mask:np.array, output_dir:str) -> None: 
        self.three_d_image = three_d_image
        self.three_d_binary_mask = three_d_binary_mask
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
        
        flow_field = self._init_flow_field()
        rs = np.linspace(r1, r2, self.z_flow)
        thetas = np.linspace(theta1, theta2, self.z_flow)
        flow_field = self._create_flow_for_r_and_scale(flow_field, rs, thetas)
        self.flow_field = self._stretch_flow_vertically(flow_field)
        self.flow_field_rotated = self._rotate_flow(flow_field, np.linalg.inv(self.main_rotation_matrix.T))
        self.flow_for_mask = self._crop_flow_by_mask_center(self.flow_field_rotated, self.orig_vertices_mean)
        self.flow_for_mask = self._interp_to_fill_nans(self.flow_for_mask)

    def handle_outside_mask(self):
        mask = self.skewed_three_d_binary_mask.astype(bool)
        
        if self.blur_around_mask_radious > 0:
            mask_dilated = ndimage.binary_dilation(mask, iterations=self.blur_around_mask_radious)
            mask_surrounding = mask_dilated^mask
            out_mask = ~mask_dilated
        else:
            out_mask = ~mask
        self.scaled_flow_for_mask[out_mask.nonzero()] = 0.
        self.scaled_flow_for_mask[mask_surrounding.nonzero()] *= 0.5 #TODO maybe put nans and then interp them to make a smoother transition.
        return self.scaled_flow_for_mask

    def flow_warp(self, image:np.array, flow:np.array, warping_borders_pad:str, warping_interp_mode:str)->np.array: #TODO move to flow utils package
        flow = np.rollaxis(flow,-1)
        flow = torch.tensor(flow)
        flow = torch.unsqueeze(flow,0)
        image = torch.tensor(image)
        image = torch.unsqueeze(torch.unsqueeze(image,0),0)

        B, _, H, W, D = flow.size()
        flow = torch.flip(flow, [1]) # flow is now z, y, x
        base_grid = self._mesh_grid(B, H, W, D).type_as(image)  # B2HW
        grid_plus_flow = base_grid + flow
        v_grid = self._norm_grid(grid_plus_flow)  # BHW2
        image_warped = nn.functional.grid_sample(image, v_grid, mode=warping_interp_mode, padding_mode=warping_borders_pad, align_corners=False)

        return image_warped[0,0,:,:,:].cpu().numpy()

    def save_nrrds(self, suffix:str)->None:
        nrrd.write(os.path.join(self.output_dir, f'image_orig{suffix}.nrrd'),   self.three_d_image)
        nrrd.write(os.path.join(self.output_dir, f'image_skewed{suffix}.nrrd'), self.skewed_three_d_image)
        nrrd.write(os.path.join(self.output_dir, f'mask_orig{suffix}.nrrd'),    self.three_d_binary_mask.astype(float))
        nrrd.write(os.path.join(self.output_dir, f'mask_skewed{suffix}.nrrd'),  self.skewed_three_d_binary_mask.astype(float))
       
    def save_npys(self, suffix:str)->None:
        np.save(os.path.join(self.output_dir, f'image_orig{suffix}.npy'),     self.three_d_image)
        np.save(os.path.join(self.output_dir, f'image_skewed{suffix}.npy'),   self.skewed_three_d_image)
        np.save(os.path.join(self.output_dir, f"flow_for_mask{suffix}.npy"),  self.scaled_flow_for_mask)
        np.save(os.path.join(self.output_dir, f"flow_for_image{suffix}.npy"), self.scaled_flow_for_image)
        np.save(os.path.join(self.output_dir, f'mask_orig{suffix}.npy'),      self.three_d_binary_mask.astype(bool))
        np.save(os.path.join(self.output_dir, f'mask_skewed{suffix}.npy'),    self.skewed_three_d_binary_mask.astype(bool))

    def _interp_to_fill_nans(self, flow) -> None:
        patchify_step = 8
        patch_size_x, patch_size_y = 10, 10
        unpatchify_output_x = flow.shape[0] - (flow.shape[0] - patch_size_x) % patchify_step
        unpatchify_output_y = flow.shape[1] - (flow.shape[1] - patch_size_y) % patchify_step

        for axis in range(3):
            for z_plane_i in range(flow.shape[2]):
                z_plane = flow[:,:,z_plane_i,axis]
                patches = patchify.patchify(z_plane, (patch_size_x, patch_size_y), step=patchify_step)    
                flow = self._interp_in_patches(flow, patches, axis, z_plane_i, unpatchify_output_x, unpatchify_output_y)
                unpatchify_dim_matches_scan_dim = (flow.shape[:2] == (unpatchify_output_x, unpatchify_output_y))
                if not(unpatchify_dim_matches_scan_dim):
                    flow[unpatchify_output_x-2:, :, z_plane_i, axis] = self._interp_missing_values(flow[unpatchify_output_x-2:, :, z_plane_i, axis], interpolator=LinearNDInterpolator)
                    flow[:, unpatchify_output_y-2:, z_plane_i, axis] = self._interp_missing_values(flow[:, unpatchify_output_y-2:, z_plane_i, axis], interpolator=LinearNDInterpolator)

        return flow

    def _interp_in_patches(self, flow:np.array, patches:np.array, axis:int, z_plane_i:int, unpatchify_output_x:int, unpatchify_output_y:int) -> None:
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch_with_nans = patches[i, j]
                patch_wo_nans = self._interp_missing_values(patch_with_nans, interpolator=LinearNDInterpolator)
                patches[i, j] = patch_wo_nans
        flow_for_axis = patchify.unpatchify(patches, (unpatchify_output_x, unpatchify_output_y))
        flow[:unpatchify_output_x,:unpatchify_output_y, z_plane_i, axis] = flow_for_axis
        return flow

    def _crop_flow_by_mask_center(self, flow_field_rotated:np.array, orig_vertices_mean:np.array) -> np.array:
        start = (2*self.center - orig_vertices_mean).astype(int)
        end = (2*self.center - orig_vertices_mean + np.array((self.x, self.y, self.z))).astype(int)
        flow_field_cropped = flow_field_rotated[ start[0,0]:end[0,0], start[0,1]:end[0,1], start[0,2]:end[0,2], : ]
        return flow_field_cropped

    def _compute_main_rotation(self)->np.array:
        shell = three_d_data_manager.extract_segmentation_envelope(self.three_d_binary_mask)
        self.orig_vertices =np.array(shell.nonzero()).T
        self.orig_vertices_mean = self.orig_vertices.mean(axis=0)
        orig_vertices_centered = self.orig_vertices - self.orig_vertices_mean
        
        U, S, Vt = np.linalg.svd(orig_vertices_centered)
        x_to_z = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        return Vt @ x_to_z

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
            edges_coords_centered = edges_coords - self.center_flow # z axis doesnt need to be reduced but it doesnt matter since the rotation is around z axis
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
            flow_field[:, :, xy_plane_i, 0] = self._interp_missing_values(flow_field[:, :, xy_plane_i, 0].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 1] = self._interp_missing_values(flow_field[:, :, xy_plane_i, 1].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 2] = self._interp_missing_values(flow_field[:, :, xy_plane_i, 2].copy(), LinearNDInterpolator)
        return flow_field
    
    def _interp_missing_values(self, flow_field_axis:np.array, interpolator)->np.array:
        nan_indices = np.isnan(flow_field_axis)
        main_points_indices = np.logical_not(nan_indices)
        main_points_data = flow_field_axis[main_points_indices]
        if main_points_data.shape[0] == 0:
            flow_field_axis[nan_indices] = 0.
        elif main_points_data.shape[0] < 3:
            flow_field_axis[nan_indices] = main_points_data.mean()
        else:
            try: 
                interp = interpolator(list(zip(*main_points_indices.nonzero())), main_points_data) 
                flow_field_axis[nan_indices] = interp(*nan_indices.nonzero())
            except QhullError as e:
                flow_field_axis[nan_indices] = main_points_data.mean()
                print(f"Can't interpolate to fill nans: \n{e}")

        return flow_field_axis

    def _stretch_flow_vertically(self, flow_field:np.array)->np.array:
        z0_to_center = np.linspace((-(self.z_flow/2) * self.h) + (self.z_flow/2), 0, self.z_flow//2)
        flow_field[:, :, :self.z_flow//2, 2] = z0_to_center
        flow_field[:, :, self.z_flow//2:, 2] = -z0_to_center[::-1]
        return flow_field

    def _rotate_flow(self, flow_field:np.array, rotation_matrix:np.array)->np.array:
        xx, yy, zz = np.meshgrid(np.arange(self.x_flow),
                                np.arange(self.y_flow),
                                np.arange(self.z_flow), indexing='ij')
        coords = np.stack(
            (xx-self.center_flow[0,0], yy-self.center_flow[0,1], zz-self.center_flow[0,2]),
            axis=-1
            ) 
        rot_coords = self._rotate_coords(coords, rotation_matrix)

        valid_indices, valid_coords = self._get_valid_coords_and_indices(rot_coords)
        valid_flow_vals = flow_field[xx, yy, zz][valid_indices]

        flow_field_rotated = self._rotate_flow_vals(valid_coords, valid_flow_vals, rotation_matrix)

        return flow_field_rotated
        
    def _rotate_coords(self, coords:np.array, rotation_matrix:np.array)->np.array:
        rot_coords = np.dot(coords.reshape((-1, 3)), rotation_matrix.T)
        rot_coords = rot_coords.reshape(self.x_flow, self.y_flow, self.z_flow, 3)
        rot_coords[:, :, :, 0] += self.center_flow[0, 0]
        rot_coords[:, :, :, 1] += self.center_flow[0, 1]
        rot_coords[:, :, :, 2] += self.center_flow[0, 2]
        return rot_coords

    def _get_valid_coords_and_indices(self, rot_coords:np.array)->Tuple[np.array,np.array]:
        valid_indices = np.all((rot_coords >= 0) & (rot_coords+0.5 < (self.x_flow, self.y_flow, self.z_flow)), axis=-1)
        valid_coords = np.round(rot_coords[valid_indices]).astype(int) 
        return valid_indices, valid_coords

    def _rotate_flow_vals(self, valid_coords:np.array, valid_flow_vals:np.array, rotation_matrix:np.array)->np.array:
        flow_field_rotated = np.empty([self.x_flow, self.y_flow, self.z_flow, 3])  
        flow_field_rotated[:] = np.nan
        flow_field_rotated[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = (np.dot(valid_flow_vals, rotation_matrix.T))
        return flow_field_rotated

    def _mesh_grid(self, B:int, H:int, W:int, D:int)->np.array: #TODO move to flow utils package
        # batches not implented
        x = torch.arange(H)
        y = torch.arange(W)
        z = torch.arange(D)
        mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0) 

        mesh = mesh.unsqueeze(0)
        return mesh.repeat([B,1,1,1,1])

    def _norm_grid(self, v_grid:np.array)->np.array: #TODO move to flow utils package
        """scale grid to [-1,1]"""
        _, _, H, W, D = v_grid.size()
        v_grid_norm = torch.zeros_like(v_grid)
        v_grid_norm[:, 0, :, :, :] = (2.0 * v_grid[:, 0, :, :, :] / (D - 1)) - 1.0 
        v_grid_norm[:, 1, :, :, :] = (2.0 * v_grid[:, 1, :, :, :] / (W - 1)) - 1.0
        v_grid_norm[:, 2, :, :, :] = (2.0 * v_grid[:, 2, :, :, :] / (H - 1)) - 1.0 
        
        return v_grid_norm.permute(0, 2, 3, 4, 1)
