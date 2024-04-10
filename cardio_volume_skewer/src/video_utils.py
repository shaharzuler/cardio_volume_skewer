from typing import List
import os

import numpy as np
import cv2


def create_video_from_xys_seqs(x_seq:List[np.array], y_seq:List[np.array], z_seq:List[np.array], filename:str, crop_pad:int=0, gap_bet_images=16) -> None:
    print(f"Saving video {filename}")
    frameSize = max(z_seq[0].shape[0], x_seq[0].shape[0]) + gap_bet_images + y_seq[0].shape[0], z_seq[0].shape[1] + gap_bet_images + x_seq[0].shape[1]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (frameSize[1],frameSize[0]))
    max_val = max(np.array(x_seq).max(), np.array(y_seq).max(), np.array(z_seq).max())
    min_val = min(np.array(x_seq).min(), np.array(y_seq).min(), np.array(z_seq).min())
    for frame_x, frame_y, frame_z in zip(x_seq, y_seq, z_seq):

        frame = np.zeros(frameSize)

        frame_x_norm = ( ( (frame_x - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)
        frame_y_norm = ( ( (frame_y - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)
        frame_z_norm = ( ( (frame_z - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)

        frame[                                   : frame_z.shape[0]                                    ,                                   : frame_z.shape[1]                                     ] = frame_z_norm
        frame[                                   : frame_x.shape[0]                                    , frame_z.shape[1] + gap_bet_images : frame_z.shape[1] + gap_bet_images + frame_x.shape[1] ] = frame_x_norm
        frame[ frame_z.shape[0] + gap_bet_images : frame_z.shape[0] + gap_bet_images + frame_y.shape[0], frame_z.shape[1] + gap_bet_images : frame_z.shape[1] + gap_bet_images + frame_y.shape[1] ] = frame_y_norm

        frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
        out.write(frame)

    out.release()

    if crop_pad>0:
        frameSize = max(z_seq[0].shape[0],x_seq[0].shape[0]) + gap_bet_images + y_seq[0].shape[0] - 4 * crop_pad, z_seq[0].shape[1] + gap_bet_images + x_seq[0].shape[1] - 4 * crop_pad
        out = cv2.VideoWriter("vid_crop.avi", cv2.VideoWriter_fourcc(*'XVID'), 10, (frameSize[1],frameSize[0]))
        max_val = max(np.array(x_seq).max(), np.array(y_seq).max(), np.array(z_seq).max())
        min_val = min(np.array(x_seq).min(), np.array(y_seq).min(), np.array(z_seq).min())
        for frame_x, frame_y, frame_z in zip(x_seq, y_seq, z_seq):

            frame = np.zeros(frameSize)

            frame_x_norm = ( ( (frame_x - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]
            frame_y_norm = ( ( (frame_x - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]
            frame_z_norm = ( ( (frame_x - min_val) / (max_val - min_val) ) * 255).astype(np.uint8)[crop_pad:-crop_pad, crop_pad:-crop_pad]

            frame[                                        : frame_z_norm.shape[0]                                         ,                                        : frame_z_norm.shape[1]                                          ] = frame_z_norm
            frame[                                        : frame_x_norm.shape[0]                                         , frame_z_norm.shape[1] + gap_bet_images : frame_z_norm.shape[1] + gap_bet_images + frame_x_norm.shape[1] ] = frame_x_norm
            frame[ frame_z_norm.shape[0] + gap_bet_images : frame_z_norm.shape[0] + gap_bet_images + frame_y_norm.shape[0], frame_z_norm.shape[1] + gap_bet_images : frame_z_norm.shape[1] + gap_bet_images + frame_y_norm.shape[1] ] = frame_y_norm

            frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
            out.write(frame)

        out.release()

def create_video_from_4d_arr(patient_4d_scan_arr, output_dir, filename):
    print(f"Saving video {filename}")
    frameSize = patient_4d_scan_arr.shape[2]+70+2*patient_4d_scan_arr.shape[1], patient_4d_scan_arr.shape[3]
    out = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc(*'XVID'), 15, (frameSize[1],frameSize[0]))
    for timestep in range(patient_4d_scan_arr.shape[0]):
        frame_x = ((patient_4d_scan_arr[timestep,patient_4d_scan_arr.shape[1]//2,:,:]/patient_4d_scan_arr.max())*255).astype(np.uint8)
        frame_y = ((patient_4d_scan_arr[timestep,:,patient_4d_scan_arr.shape[2]//2,:]/patient_4d_scan_arr.max())*255).astype(np.uint8)
        frame_z = ((patient_4d_scan_arr[timestep,:,:,patient_4d_scan_arr.shape[3]//2]/patient_4d_scan_arr.max())*255).astype(np.uint8)

        frame = np.vstack([frame_x,  np.zeros((10,256)),frame_y,np.zeros((10,256)),frame_z, np.zeros((50,256))])

        frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_BONE)
        out.write(frame)

    out.release()
