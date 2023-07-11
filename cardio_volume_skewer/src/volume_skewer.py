from typing import Tuple
import numpy as np
from scipy import ndimage
import math
from scipy.spatial.transform import Rotation as R
import nrrd
from scipy.interpolate import LinearNDInterpolator
import torch
from torch import nn
import patchify
import os




class VolumeSkewer:
    def __init__(self, save_nrrd:bool=True, save_npy:bool=True, zero_outside_mask:bool=True, warping_borders_pad='zeros', img_warping_interp_mode='bilinear', mask_warping_interp_mode='nearest'):
        self.save_nrrd = save_nrrd
        self.save_npy = save_npy
        self.zero_outside_mask = zero_outside_mask
        self.warping_borders_pad=warping_borders_pad
        self.img_warping_interp_mode = img_warping_interp_mode
        self.mask_warping_interp_mode = mask_warping_interp_mode
        
    def skew_volume(self,  theta1:float, theta2:float, r1:float, r2:float, h:float, three_d_image:np.array, three_d_binary_mask:np.array, output_dir:str) -> np.array:
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
        self.main_rotation_matrix = self.compute_main_rotation()

        self.x_flow = 2 * self.x
        self.y_flow = 2 * self.y
        self.z_flow = 2 * self.z
        self.center_flow = 2 * self.center
        
        flow_field = self.init_flow_field()
        rs = np.linspace(r1, r2, self.z_flow)
        thetas = np.linspace(theta1, theta2, self.z_flow)
        flow_field = self.create_flow_for_r_and_scale(flow_field, rs, thetas)
        self.flow_field = self.stretch_flow_vertically(flow_field)
        self.flow_field_rotated = self.rotate_flow(flow_field, np.linalg.inv(self.main_rotation_matrix.T))
        self.cropped_flow = self.crop_flow_by_mask_center(self.flow_field_rotated, self.orig_vertices_mean)

        if self.zero_outside_mask: # only moves pixels inside the seg mask
            out_mask = ~self.three_d_binary_mask.astype(bool)
            self.cropped_flow[out_mask.nonzero()] = 0.

            mask_ind = self.three_d_binary_mask.nonzero()
            min_ind = np.array((mask_ind[0].min(), mask_ind[1].min(), mask_ind[2].min()), dtype=int) - 10
            max_ind = np.array((mask_ind[0].max(), mask_ind[1].max(), mask_ind[2].max()), dtype=int) + 10
        
            cropped_flow_around_LV = self.cropped_flow[min_ind[0]:max_ind[0], min_ind[1]:max_ind[1], min_ind[2]:max_ind[2], :]
            cropped_flow_around_LV = self.interp_to_fill_nans(cropped_flow_around_LV)
            self.cropped_flow[min_ind[0]:max_ind[0], min_ind[1]:max_ind[1], min_ind[2]:max_ind[2], :] = cropped_flow_around_LV

        else:
            self.cropped_flow = self.interp_to_fill_nans(self.cropped_flow)


        self.skewed_three_d_image       = self.flow_warp(img=self.three_d_image,                     flow=self.cropped_flow, warping_borders_pad=self.warping_borders_pad, warping_interp_mode=self.img_warping_interp_mode )
        self.skewed_three_d_binary_mask = self.flow_warp(img=self.three_d_binary_mask.astype(float), flow=self.cropped_flow, warping_borders_pad=self.warping_borders_pad, warping_interp_mode=self.mask_warping_interp_mode)

        if self.save_nrrd:
            self.save_nrrds()
        if self.save_npy:
            self.save_npys()

        return self.skewed_three_d_image

    def interp_to_fill_nans(self, flow) -> None:
        patchify_step = 8
        patch_size_x, patch_size_y = 10, 10
        unpatchify_output_x = flow.shape[0] - (flow.shape[0] - patch_size_x) % patchify_step
        unpatchify_output_y = flow.shape[1] - (flow.shape[1] - patch_size_y) % patchify_step

        for axis in range(3):
            for z_plane_i in range(flow.shape[2]):
                z_plane = flow[:,:,z_plane_i,axis]
                patches = patchify.patchify(z_plane, (patch_size_x, patch_size_y), step=patchify_step)    
                flow = self.interp_in_patches(flow, patches, axis, z_plane_i, unpatchify_output_x, unpatchify_output_y)
                unpatchify_dim_matches_scan_dim = (flow.shape[:2] == (unpatchify_output_x, unpatchify_output_y))
                if not(unpatchify_dim_matches_scan_dim):
                    flow[unpatchify_output_x-2:, :, z_plane_i, axis] = self.interp_missing_values(flow[unpatchify_output_x-2:, :, z_plane_i, axis], interpolator=LinearNDInterpolator)
                    flow[:, unpatchify_output_y-2:, z_plane_i, axis] = self.interp_missing_values(flow[:, unpatchify_output_y-2:, z_plane_i, axis], interpolator=LinearNDInterpolator)

        return flow

    def interp_in_patches(self, flow:np.array, patches:np.array, axis:int, z_plane_i:int, unpatchify_output_x:int, unpatchify_output_y:int) -> None:
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch_with_nans = patches[i, j]
                patch_wo_nans = self.interp_missing_values(patch_with_nans, interpolator=LinearNDInterpolator)
                patches[i, j] = patch_wo_nans
        flow_for_axis = patchify.unpatchify(patches, (unpatchify_output_x, unpatchify_output_y))
        flow[:unpatchify_output_x,:unpatchify_output_y, z_plane_i, axis] = flow_for_axis
        return flow

    def crop_flow_by_mask_center(self, flow_field_rotated:np.array, orig_vertices_mean:np.array) -> np.array:
        start = (2*self.center - orig_vertices_mean).astype(int)
        end = (2*self.center - orig_vertices_mean + np.array((self.x, self.y, self.z))).astype(int)
        flow_field_cropped = flow_field_rotated[ start[0,0]:end[0,0], start[0,1]:end[0,1], start[0,2]:end[0,2], : ]
        return flow_field_cropped

    def compute_main_rotation(self)->np.array:
        shell = VolumeSkewer.extract_shell_from_mask(self.three_d_binary_mask)
        self.orig_vertices =np.array(shell.nonzero()).T
        self.orig_vertices_mean = self.orig_vertices.mean(axis=0)
        orig_vertices_centered = self.orig_vertices - self.orig_vertices_mean
        
        U, S, Vt = np.linalg.svd(orig_vertices_centered)
        x_to_z = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        return Vt @ x_to_z

    def init_flow_field(self)->np.array:
        flow_field = np.empty([self.x_flow, self.y_flow, self.z_flow, 3])  
        flow_field[:] = np.nan
        return flow_field

    def create_flow_for_r_and_scale(self, flow_field:np.array, rs:np.array, thetas:np.array)->np.array:
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
            flow_field[:, :, xy_plane_i, 0] = self.interp_missing_values(flow_field[:, :, xy_plane_i, 0].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 1] = self.interp_missing_values(flow_field[:, :, xy_plane_i, 1].copy(), LinearNDInterpolator)
            flow_field[:, :, xy_plane_i, 2] = self.interp_missing_values(flow_field[:, :, xy_plane_i, 2].copy(), LinearNDInterpolator)
        return flow_field
    
    def interp_missing_values(self, flow_field_axis:np.array, interpolator)->np.array:
        nan_indices = np.isnan(flow_field_axis)
        main_points_indices = np.logical_not(nan_indices)
        main_points_data = flow_field_axis[main_points_indices]
        interp = interpolator(list(zip(*main_points_indices.nonzero())), main_points_data) 
        flow_field_axis[nan_indices] = interp(*nan_indices.nonzero())
        
        return flow_field_axis

    def stretch_flow_vertically(self, flow_field:np.array)->np.array:
        z0_to_center = np.linspace((-(self.z_flow/2) * self.h) + (self.z_flow/2), 0, self.z_flow//2)
        flow_field[:, :, :self.z_flow//2, 2] = z0_to_center
        flow_field[:, :, self.z_flow//2:, 2] = -z0_to_center[::-1]
        return flow_field

    def rotate_flow(self, flow_field:np.array, rotation_matrix:np.array)->np.array:
        xx, yy, zz = np.meshgrid(np.arange(self.x_flow),
                                np.arange(self.y_flow),
                                np.arange(self.z_flow), indexing='ij')
        coords = np.stack(
            (xx-self.center_flow[0,0], yy-self.center_flow[0,1], zz-self.center_flow[0,2]),
            axis=-1
            ) 
        rot_coords = self.rotate_coords(coords, rotation_matrix)

        valid_indices, valid_coords = self.get_valid_coords_and_indices(rot_coords)
        valid_flow_vals = flow_field[xx, yy, zz][valid_indices]

        flow_field_rotated = self.rotate_flow_vals(valid_coords, valid_flow_vals, rotation_matrix)

        return flow_field_rotated
        
    def rotate_coords(self, coords:np.array, rotation_matrix:np.array)->np.array:
        rot_coords = np.dot(coords.reshape((-1, 3)), rotation_matrix.T)
        rot_coords = rot_coords.reshape(self.x_flow, self.y_flow, self.z_flow, 3)
        rot_coords[:, :, :, 0] += self.center_flow[0, 0]
        rot_coords[:, :, :, 1] += self.center_flow[0, 1]
        rot_coords[:, :, :, 2] += self.center_flow[0, 2]
        return rot_coords

    def get_valid_coords_and_indices(self, rot_coords:np.array)->Tuple[np.array,np.array]:
        valid_indices = np.all((rot_coords >= 0) & (rot_coords+0.5 < (self.x_flow, self.y_flow, self.z_flow)), axis=-1)
        valid_coords = np.round(rot_coords[valid_indices]).astype(int) 
        return valid_indices, valid_coords

    def rotate_flow_vals(self, valid_coords:np.array, valid_flow_vals:np.array, rotation_matrix:np.array)->np.array:
        flow_field_rotated = np.empty([self.x_flow, self.y_flow, self.z_flow, 3])  
        flow_field_rotated[:] = np.nan
        flow_field_rotated[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = (np.dot(valid_flow_vals, rotation_matrix.T))
        return flow_field_rotated

    def flow_warp(self, img:np.array, flow:np.array, warping_borders_pad:str, warping_interp_mode:str)->np.array: #TODO move to flow utils package
        flow = np.rollaxis(flow,-1)
        flow = torch.tensor(flow)
        flow = torch.unsqueeze(flow,0)
        img = torch.tensor(img)
        img = torch.unsqueeze(torch.unsqueeze(img,0),0)

        B, _, H, W, D = flow.size()
        flow = torch.flip(flow, [1]) #flow is now z, y, x
        base_grid = self.mesh_grid(B, H, W, D).type_as(img)  # B2HW
        grid_plus_flow = base_grid + flow
        v_grid = self.norm_grid(grid_plus_flow)  # BHW2
        img_warped = nn.functional.grid_sample(img, v_grid, mode=warping_interp_mode, padding_mode=warping_borders_pad, align_corners=False)

        return img_warped[0,0,:,:,:].cpu().numpy()

    def mesh_grid(self, B:int, H:int, W:int, D:int)->np.array: #TODO move to flow utils package
        # batches not implented
        x = torch.arange(H)
        y = torch.arange(W)
        z = torch.arange(D)
        mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0) 

        mesh = mesh.unsqueeze(0)
        return mesh.repeat([B,1,1,1,1])

    def norm_grid(self, v_grid:np.array)->np.array: #TODO move to flow utils package
        """scale grid to [-1,1]"""
        _, _, H, W, D = v_grid.size()
        v_grid_norm = torch.zeros_like(v_grid)
        v_grid_norm[:, 0, :, :, :] = (2.0 * v_grid[:, 0, :, :, :] / (D - 1)) - 1.0 #(H - 1)) - 1.0
        v_grid_norm[:, 1, :, :, :] = (2.0 * v_grid[:, 1, :, :, :] / (W - 1)) - 1.0
        v_grid_norm[:, 2, :, :, :] = (2.0 * v_grid[:, 2, :, :, :] / (H - 1)) - 1.0 #(D - 1)) - 1.0
        
        return v_grid_norm.permute(0, 2, 3, 4, 1)

    def save_nrrds(self)->None:
        suffix = f"_thetas_{round(self.theta1,2)}_{round(self.theta2,2)}_rs_{round(self.r1,2)}_{round(self.r2,2)}_h_{round(self.h,2)}"
        nrrd.write(os.path.join(self.output_dir, f'img_orig{suffix}.nrrd'), self.three_d_image)
        nrrd.write(os.path.join(self.output_dir, f'img_skewed{suffix}.nrrd'),      self.skewed_three_d_image)
        nrrd.write(os.path.join(self.output_dir, f'img_diff{suffix}.nrrd'),        self.skewed_three_d_image - self.three_d_image)
        nrrd.write(os.path.join(self.output_dir, f"flow_magnitude{suffix}.nrrd"),  self.cropped_flow[:,:,:,0]**2 + self.cropped_flow[:,:,:,1]**2 + self.cropped_flow[:,:,:,2]**2)
        nrrd.write(os.path.join(self.output_dir, f'mask_orig{suffix}.nrrd'),       self.three_d_binary_mask.astype(float))
        nrrd.write(os.path.join(self.output_dir, f'mask_skewed{suffix}.nrrd'),     self.skewed_three_d_binary_mask.astype(float))
       
    def save_npys(self)->None:
        suffix = f"_thetas_{round(self.theta1,2)}_{round(self.theta2,2)}_rs_{round(self.r1,2)}_{round(self.r2,2)}_h_{round(self.h,2)}"
        np.save(os.path.join(self.output_dir, f'img_orig{suffix}.npy'), self.three_d_image)
        np.save(os.path.join(self.output_dir, f'img_skewed{suffix}.npy'),      self.skewed_three_d_image)
        np.save(os.path.join(self.output_dir, f'img_diff{suffix}.npy'),        self.skewed_three_d_image - self.three_d_image)
        np.save(os.path.join(self.output_dir, f"flow_magnitude{suffix}.npy"),  self.cropped_flow[:,:,:,0]**2 + self.cropped_flow[:,:,:,1]**2 + self.cropped_flow[:,:,:,2]**2)
        np.save(os.path.join(self.output_dir, f"flow{suffix}.npy"),            self.cropped_flow)
        np.save(os.path.join(self.output_dir, f'mask_orig{suffix}.npy'),       self.three_d_binary_mask.astype(bool))
        np.save(os.path.join(self.output_dir, f'mask_skewed{suffix}.npy'),     self.skewed_three_d_binary_mask.astype(bool))

    @staticmethod
    def extract_shell_from_mask(three_d_binary_mask:np.array) -> np.array:
        erosed_mask = ndimage.binary_erosion(three_d_binary_mask)
        three_d_shell = np.logical_xor(three_d_binary_mask, erosed_mask)
        return three_d_shell
