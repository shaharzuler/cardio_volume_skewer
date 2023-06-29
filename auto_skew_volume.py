# TODO fix opposite flow direction
# TODO automatically copy orig image to be the first image
# TODO add pad type as a param

from typing import List, Tuple
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import nrrd
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import torch
from torch import nn
import patchify
import time 
import os
import scipy
import cv2

from data_handling.fourdct_heart_data_handler import images_to_3dnump


class VolumeSkewer:
    def __init__(self, save_nrrd:bool=True, zero_outside_mask:bool=True, warping_borders_pad='zeros', warping_interp_mode='bilinear'):
        self.save_nrrd = save_nrrd
        self.zero_outside_mask = zero_outside_mask
        self.warping_borders_pad=warping_borders_pad
        self.warping_interp_mode = warping_interp_mode
        
    def skew_volume(self,  theta1:float, theta2:float, r1:float, r2:float, h:float, three_d_image:np.array, three_d_binary_mask:np.array, output_dir:str) -> np.array:
        self.three_d_image = three_d_image
        self.three_d_binary_mask = three_d_binary_mask
        self.output_dir = output_dir
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


        self.skewed_three_d_image = self.flow_warp(self.three_d_image, self.cropped_flow, warping_borders_pad=self.warping_borders_pad, warping_interp_mode=self.warping_interp_mode)
        if self.save_nrrd:
            self.save_nrrds()

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
        interp = interpolator(list(zip(*main_points_indices.nonzero())), main_points_data) # LinearNDInterpolator(list(zip(*main_points_indices.nonzero())), main_points_data)
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

    def flow_warp(self, img:np.array, flow:np.array, warping_borders_pad:str, warping_interp_mode:str)->np.array:
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

    def mesh_grid(self, B:int, H:int, W:int, D:int)->np.array:
        # batches not implented
        x = torch.arange(H)
        y = torch.arange(W)
        z = torch.arange(D)
        mesh = torch.stack(torch.meshgrid(x, y, z)[::-1], 0) # torch.stack(torch.meshgrid(x, y, z), 0) # 

        mesh = mesh.unsqueeze(0)
        return mesh.repeat([B,1,1,1,1])

    def norm_grid(self, v_grid:np.array)->np.array:
        """scale grid to [-1,1]"""
        _, _, H, W, D = v_grid.size()
        v_grid_norm = torch.zeros_like(v_grid)
        v_grid_norm[:, 0, :, :, :] = (2.0 * v_grid[:, 0, :, :, :] / (D - 1)) - 1.0 #(H - 1)) - 1.0
        v_grid_norm[:, 1, :, :, :] = (2.0 * v_grid[:, 1, :, :, :] / (W - 1)) - 1.0
        v_grid_norm[:, 2, :, :, :] = (2.0 * v_grid[:, 2, :, :, :] / (H - 1)) - 1.0 #(D - 1)) - 1.0
        
        return v_grid_norm.permute(0, 2, 3, 4, 1)

    def save_nrrds(self)->None:
        suffix = f"_thetas_{round(self.theta1,2)}_{round(self.theta2,2)}_rs_{round(self.r1,2)}_{round(self.r2,2)}_h_{round(self.h,2)}"
        nrrd.write(os.path.join(self.output_dir, f'img_orig_thetas{suffix}.nrrd'), self.three_d_image)
        nrrd.write(os.path.join(self.output_dir,f'img_skewed{suffix}.nrrd'), self.skewed_three_d_image)
        nrrd.write(os.path.join(self.output_dir,f'img_diff{suffix}.nrrd'), self.skewed_three_d_image - self.three_d_image)
        nrrd.write(os.path.join(self.output_dir,f"flow{suffix}.nrrd"),self.cropped_flow[:,:,:,0]**2 + self.cropped_flow[:,:,:,1]**2 + self.cropped_flow[:,:,:,2]**2)

    @staticmethod
    def extract_shell_from_mask(three_d_binary_mask:np.array) -> np.array:
        erosed_mask = ndimage.binary_erosion(three_d_binary_mask)
        three_d_shell = np.logical_xor(three_d_binary_mask, erosed_mask)
        return three_d_shell

class VideoUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_video_from_xys_seqs(x_seq:List[np.array], y_seq:List[np.array], z_seq:List[np.array], filename:str, crop_pad:int=0, gap_bet_images=16) -> None:
        print(f"Saving video {filename}")
        frameSize = z_seq[0].shape[0] + gap_bet_images + y_seq[0].shape[0], z_seq[0].shape[1] + gap_bet_images + x_seq[0].shape[1]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (frameSize[1],frameSize[0]))
        max_val = max(np.array(x_seq).max(), np.array(y_seq).max(), np.array(z_seq).max())
        for frame_x, frame_y, frame_z in zip(x_seq, y_seq, z_seq):

            frame = np.zeros(frameSize)

            frame_x_norm = ( (frame_x / max_val) * 255).astype(np.uint8)
            frame_y_norm = ( (frame_y / max_val) * 255).astype(np.uint8)
            frame_z_norm = ( (frame_z / max_val) * 255).astype(np.uint8)

            frame[                                   : frame_z.shape[0]                                    ,                                   : frame_z.shape[1]                                     ] = frame_z_norm
            frame[                                   : frame_z.shape[0]                                    , frame_z.shape[1] + gap_bet_images : frame_z.shape[1] + gap_bet_images + frame_x.shape[1] ] = frame_x_norm
            frame[ frame_z.shape[0] + gap_bet_images : frame_z.shape[0] + gap_bet_images + frame_y.shape[0], frame_z.shape[1] + gap_bet_images : frame_z.shape[1] + gap_bet_images + frame_y.shape[1] ] = frame_y_norm

            frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
            out.write(frame)

        out.release()

        if crop_pad>0:
            frameSize = z_seq[0].shape[0] + gap_bet_images + y_seq[0].shape[0] - 4 * crop_pad, z_seq[0].shape[1] + gap_bet_images + x_seq[0].shape[1] - 4 * crop_pad
            out = cv2.VideoWriter("vid_crop.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (frameSize[1],frameSize[0]))
            max_val = max(np.array(x_seq).max(), np.array(y_seq).max(), np.array(z_seq).max())
            for frame_x, frame_y, frame_z in zip(x_seq, y_seq, z_seq):

                frame = np.zeros(frameSize)

                frame_x_norm = ( (frame_x / max_val) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]
                frame_y_norm = ( (frame_y / max_val) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]
                frame_z_norm = ( (frame_z / max_val) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]

                frame[                                        : frame_z_norm.shape[0]                                         ,                                        : frame_z_norm.shape[1]                                          ] = frame_z_norm
                frame[                                        : frame_z_norm.shape[0]                                         , frame_z_norm.shape[1] + gap_bet_images : frame_z_norm.shape[1] + gap_bet_images + frame_x_norm.shape[1] ] = frame_x_norm
                frame[ frame_z_norm.shape[0] + gap_bet_images : frame_z_norm.shape[0] + gap_bet_images + frame_y_norm.shape[0], frame_z_norm.shape[1] + gap_bet_images : frame_z_norm.shape[1] + gap_bet_images + frame_y_norm.shape[1] ] = frame_y_norm

                frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
                out.write(frame)

            out.release()

    @staticmethod
    def create_video_from_4d_arr(patient_4d_scan_arr, output_dir, filename):
        print(f"Saving video {filename}")
        frameSize = patient_4d_scan_arr.shape[2]+70+2*patient_4d_scan_arr.shape[1], patient_4d_scan_arr.shape[3]
        out = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc(*'XVID'), 15, (frameSize[1],frameSize[0]))
        for timestep in range(patient_4d_scan_arr.shape[0]):
            # frame = ((patient_4d_scan_arr[timestep,patient_4d_scan_arr.shape[1]//2,:,:]/patient_4d_scan_arr.max())*255).astype(np.uint8)

            frame_x = ((patient_4d_scan_arr[timestep,patient_4d_scan_arr.shape[1]//2,:,:]/patient_4d_scan_arr.max())*255).astype(np.uint8)
            frame_y = ((patient_4d_scan_arr[timestep,:,patient_4d_scan_arr.shape[2]//2,:]/patient_4d_scan_arr.max())*255).astype(np.uint8)
            frame_z = ((patient_4d_scan_arr[timestep,:,:,patient_4d_scan_arr.shape[3]//2]/patient_4d_scan_arr.max())*255).astype(np.uint8)

            frame = np.vstack([frame_x,  np.zeros((10,256)),frame_y,np.zeros((10,256)),frame_z, np.zeros((50,256))])

            frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
            out.write(frame)

        out.release()

def create_toy_box_image_and_mask():
    binary_mask = np.zeros((100,100,100), dtype=bool)
    binary_mask[40:60, 40:60, 10:90] = True
    binary_mask[30:50, 30:50, 10:40] = True
    binary_mask[50:70, 50:70, 40:60] = True
    binary_mask[70:90, 70:90, 60:90] = True
    binary_mask[20:81,40:61, 40:61] = True


    three_d_image = np.arange(binary_mask.flatten().shape[0]).astype(float)
    three_d_image = three_d_image.reshape(binary_mask.shape)
    three_d_image = binary_mask.copy().astype(float)

def read_ct_and_mask(timestep, save_nrrd=False):
    ct_dir = os.path.join("/","home","shahar","projects","4dct_data","20","20","Anonymized - 859733","Ctacoc","DS_CorCTABi 1.5 B25f 0-95% Matrix 256 - 12")

    img3d, voxel_size = images_to_3dnump(ct_dir, img_num=timestep, plot=False)
    img3d_scaled = scipy.ndimage.zoom(img3d, voxel_size)
    voxel_size = np.array([0.78125, 0.78125, 1.5    ])

    mask_path = os.path.join("/","home","shahar","projects","flow","_4DCTCostUnrolling-main","warped_seg_maps2", "smoothening_exp", "seg_20_28to28.npz")
    mask_path = f"/home/shahar/projects/pcd_to_mesh/exploration/binary_mask_{timestep}.npz"
    try:
        mask = np.load(mask_path)["arr_0"]
        mask_xyz = np.rollaxis(mask, 0, 3)
        mask_scaled = scipy.ndimage.zoom(mask_xyz, voxel_size, order=0, mode="nearest") 
    except FileNotFoundError:
        mask_scaled = np.load(mask_path.replace(".npz", ".npy"))
    print(img3d_scaled.shape, mask_scaled.shape)
    
    if save_nrrd:
        np.save("ct_scan", three_d_image)
        np.save("binary_mask", binary_mask)

    return img3d_scaled.copy(), mask_scaled.copy()

def create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, nrrds_dir):
    x_seq = []
    y_seq = []
    z_seq = []
    for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}"
        nrrd_arr = nrrd.read(os.path.join(nrrds_dir, f"img_skewed{suffix}.nrrd"))[0]
        x_seq.append(nrrd_arr[nrrd_arr.shape[0]//2,:,:])
        y_seq.append(nrrd_arr[:,nrrd_arr.shape[1]//2,:])
        z_seq.append(nrrd_arr[:,:,nrrd_arr.shape[2]//2])
    for r1, r2, theta1, theta2, h in zip(r1s[1:-1][::-1], r2s[1:-1][::-1], theta1s[1:-1][::-1], theta2s[1:-1][::-1], hs[1:-1][::-1]):
        print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
        suffix = f"_thetas_{round(theta1,2)}_{round(theta2,2)}_rs_{round(r1,2)}_{round(r2,2)}_h_{round(h,2)}"
        nrrd_arr = nrrd.read(os.path.join(nrrds_dir, f"img_skewed{suffix}.nrrd"))[0]
        x_seq.append(nrrd_arr[nrrd_arr.shape[0]//2,:,:])
        y_seq.append(nrrd_arr[:,nrrd_arr.shape[1]//2,:])
        z_seq.append(nrrd_arr[:,:,nrrd_arr.shape[2]//2])
    return x_seq, y_seq, z_seq

def calc_start(orig_param_start, end, num_frames):
    gap = (end - orig_param_start)/num_frames
    return orig_param_start + gap


def create_skewed_sequences(r1s_end, r2s_end, theta1s_end, theta2s_end, hs_end):
    timestep = 18 if r1s_end<=1 else 28

    num_frames = 5

    r1s_start = 1.0
    r2s_start = 1.0
    theta1s_start = 0
    theta2s_start = 0
    hs_start = 1.0
    
    r1s     = np.linspace( r1s_start,     r1s_end,     num_frames + 1 )
    r2s     = np.linspace( r2s_start,     r2s_end,     num_frames + 1 )
    theta1s = np.linspace( theta1s_start, theta1s_end, num_frames + 1 )
    theta2s = np.linspace( theta2s_start, theta2s_end, num_frames + 1 )
    hs      = np.linspace( hs_start,      hs_end,      num_frames + 1 )

    major_output_dir = "self_validation_params_exp"
    os.makedirs(major_output_dir, exist_ok=True)
    minor_output_dir = f"_thetas_{round(theta1s_end,2)}_{round(theta2s_end,2)}_rs_{round(r1s_end,2)}_{round(r2s_end,2)}_h_{round(hs_end,2)}"
    output_dir = os.path.join(major_output_dir, minor_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    SKEW = True


    if SKEW:
        three_d_image, binary_mask = read_ct_and_mask(timestep=timestep)
        volume_skewer =  VolumeSkewer(warping_borders_pad='zeros', warping_interp_mode='bilinear')
        for r1, r2, theta1, theta2, h in zip(r1s, r2s, theta1s, theta2s, hs):
            print( round(theta1, 2), round(theta2, 2), round(r1, 2), round(r2, 2), round(h, 2) )
            skewed_3d_image = volume_skewer.skew_volume(
                theta1=theta1, theta2=theta2, 
                r1=r1, r2=r2, 
                h=h, 
                three_d_image=three_d_image, 
                three_d_binary_mask=binary_mask,
                output_dir=output_dir
                )
        print("skewing completed")

    x_seq, y_seq, z_seq = create_frame_sequences_for_video(r1s, r2s, theta1s, theta2s, hs, output_dir)

    VideoUtils.create_video_from_xys_seqs(x_seq*10, y_seq*10, z_seq*10, os.path.join(output_dir,f"vid{minor_output_dir}.avi"))


## WINNER!!! create_skewed_sequences(r1s_end=0.5,r2s_end=0.5, theta1s_end=40, theta2s_end=-40, hs_end=0.8 ) 

create_skewed_sequences(r1s_end=0.8, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=0.6, r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=0.8, theta1s_end=0., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=0.6, theta1s_end=0., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=30., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=45., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=60., theta2s_end=-0., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-30., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-45., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-60., hs_end=1.) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.8) 
create_skewed_sequences(r1s_end=1., r2s_end=1., theta1s_end=0., theta2s_end=-0., hs_end=0.6) 



