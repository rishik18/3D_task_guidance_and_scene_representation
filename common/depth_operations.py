import os
import json
import argparse
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from common import constants
from models.smpl import SMPL    
from smplify import SMPLify  
from losses import perspective_projection  
from pytorch3d import transforms
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full, full2crop_cam
import time
from common.imutils import process_image 
from common.scene_operations import rgb_hsv_mask, centroid_px, add_reference_frame, add_prob_centroid2scene


def px_to_cam(p, depth_m, K):
    """p = (u,v)-tuple, depth_m = float32 array in metres, returns [X,Y,Z] in camera frame"""
    u, v = int(p[0]), int(p[1])
    z = float(depth_m[v, u])
    if z <= 0:                                     # invalid / missing depth
        return None
    #z*=10
    #z=z+0.15
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    return np.array([x, y, z], dtype=np.float32)

def get_pelvis_translation(pelvis_xy, depth_m, K,device="cuda"):         
    u_pelvis, v_pelvis = pelvis_xy.astype(int)     
    pelvis_depth = depth_m[v_pelvis, u_pelvis]     
    pelvis_3d_coord = px_to_cam([u_pelvis, v_pelvis], depth_m, K)
    pelv_x, pelv_y, pelv_z =pelvis_3d_coord
    pelvis_translation = torch.tensor([[pelv_x, pelv_y, pelv_z]], dtype=torch.float32, device=device)
    return pelvis_translation

def get_probe_centroid(frame, depth_m, K):
    mask        = rgb_hsv_mask(frame)
    centroid_uv = centroid_px(mask)
    centroid_3d = None
    if centroid_uv is not None:
        centroid_3d = px_to_cam(centroid_uv, depth_m, K)
    return centroid_3d

def depth_scaled_metric(depth_u8, near=0.5, far=5.46, offset=0.3):
    depth_m  = (depth_u8.astype(np.float32) / 255.0) * (far - near) + near
    depth_m += offset
    invalid  = depth_u8 == 0
    depth_m[invalid] = 0.0
    return depth_m

def smpl_fix_coordinates(vertices, joints, pelvis_translation):
    
    if vertices.dim() == 2:          # (6890,3) → (1,6890,3)
        vertices = vertices.unsqueeze(0)
    if joints.dim() == 2:            # (J,3)    → (1,J,3)
        joints = joints.unsqueeze(0)
    if pelvis_translation.dim() == 1:    # (3,) → (1,3)
        pelvis_translation = pelvis_translation.unsqueeze(0)

    jpelv = joints[0,8].unsqueeze(0)

    #print(f"jpelv :{jpelv}")
    #print(f"pelvis_translation: {pelvis_translation}")
    pelvis_zero_vertices  = vertices - jpelv[:, None, :]   # (B,6890,3)
    pelvis_zero_joints    = joints   - jpelv[:, None, :] 

    new_opt_vertices  = pelvis_zero_vertices + pelvis_translation[:, None, :]  #-data_full_cam[:, None, :] # (B,6890,3)
    new_joints = pelvis_zero_joints   + pelvis_translation[:, None, :] #-data_full_cam[:, None, :]
    return new_opt_vertices, new_joints

def project_cam_to_px(XYZ, K):
    """Project camera-frame point XYZ → pixel (u,v)."""
    X, Y, Z = XYZ
    u = (K[0,0] * X) / Z + K[0,2]
    v = (K[1,1] * Y) / Z + K[1,2]
    return int(round(u)), int(round(v)), Z