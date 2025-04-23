
# -*- coding: utf-8 -*-

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
from common.utils import strip_prefix_if_present, cam_crop2full

# SMPL expected joint ordering as provided
JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Kinect joint ordering (indices) and their names are defined as:
#   0: PELVIS
#   1: SPINE_NAVEL
#   2: SPINE_CHEST
#   3: NECK
#   4: CLAVICLE_LEFT
#   5: SHOULDER_LEFT
#   6: ELBOW_LEFT
#   7: WRIST_LEFT
#   8: HAND_LEFT
#   9: HANDTIP_LEFT
#   10: THUMB_LEFT
#   11: CLAVICLE_RIGHT
#   12: SHOULDER_RIGHT
#   13: ELBOW_RIGHT
#   14: WRIST_RIGHT
#   15: HAND_RIGHT
#   16: HANDTIP_RIGHT
#   17: THUMB_RIGHT
#   18: HIP_LEFT
#   19: KNEE_LEFT
#   20: ANKLE_LEFT
#   21: FOOT_LEFT
#   22: HIP_RIGHT
#   23: KNEE_RIGHT
#   24: ANKLE_RIGHT
#   25: FOOT_RIGHT
#   26: HEAD
#   27: NOSE
#   28: EYE_LEFT
#   29: EAR_LEFT
#   30: EYE_RIGHT
#   31: EAR_RIGHT

# Create a mapping dictionary from the SMPL joint names to the Kinect indices.
# For joints not directly available (or needing a combination), we provide special instructions.
joint_mapping = {
    # OpenPose joints
    'OP Nose': 27,
    'OP Neck': 3,
    'OP RShoulder': 12,
    'OP RElbow': 13,
    'OP RWrist': 14,
    'OP LShoulder': 5,
    'OP LElbow': 6,
    'OP LWrist': 7,
    # For OP MidHip, use the average of the left and right hips (indices 18 and 22)
    'OP MidHip': 0,
    'OP RHip': 22,
    'OP RKnee': 23,
    'OP RAnkle': 24,
    'OP LHip': 18,
    'OP LKnee': 19,
    'OP LAnkle': 20,
    'OP REye': 30,   # Kinect's EYE_RIGHT
    'OP LEye': 28,   # Kinect's EYE_LEFT
    'OP REar': 31,   # Kinect's EAR_RIGHT
    'OP LEar': 29,   # Kinect's EAR_LEFT
    # The following joints are not provided by Kinect.
    'OP LBigToe': 21,
    'OP LSmallToe': None,
    'OP LHeel': None,
    'OP RBigToe': 25,
    'OP RSmallToe': None,
    'OP RHeel': None,
    # Ground Truth joints
    'Right Ankle': 24,
    'Right Knee': 23,
    'Right Hip': 22,
    'Left Hip': 18,
    'Left Knee': 19,
    'Left Ankle': 20,
    'Right Wrist': 14,
    'Right Elbow': 13,
    'Right Shoulder': 12,
    'Left Shoulder': 5,
    'Left Elbow': 6,
    'Left Wrist': 7,
    'Neck (LSP)': 3,
    'Top of Head (LSP)': 26,
    'Pelvis (MPII)': 0,
    'Thorax (MPII)': 2,  # Using SPINE_CHEST
    'Spine (H36M)': 1,   # Using SPINE_NAVEL
    'Jaw (H36M)': None,  # Not provided by Kinect
    'Head (H36M)': 26,
    'Nose': 27,
    'Left Eye': 28,
    'Right Eye': 30,
    'Left Ear': 29,
    'Right Ear': 31
}

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the input frame (BGR) to a normalized tensor.
    Returns norm_img as a tensor of shape (1, 3, H, W) in float32.
    """
    resized = cv2.resize(frame, target_size)
    rgb = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5  
    #rgb = resized/255.0 #cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    #cv2.imwrite("norm_img.jpg", rgb*255)
    norm_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return norm_img


def map_kinect_to_smpl(kinect_keypoints):
    """
    Map Kinect keypoints to SMPL joint ordering.
    
    Parameters:
      kinect_keypoints (np.array): Array of shape (32, D) where D is the dimensionality of each joint (e.g. 2 for 2D, 3 for 3D).
      
    Returns:
      np.array: An array of shape (49, D) arranged in the SMPL joint order.
              For joints not available from Kinect, a zero vector is inserted.
    """
    # Determine the dimensionality (e.g., 2D or 3D) from the Kinect keypoints.
    num_dims = kinect_keypoints.shape[1] if len(kinect_keypoints.shape) > 1 else 1
    smpl_keypoints = []
    
    for joint_name in JOINT_NAMES:
        mapping = joint_mapping[joint_name]
        if mapping is None:
            # Joint not available: fill with zeros.
            smpl_keypoints.append(np.zeros(num_dims))
        else:
            smpl_keypoints.append(kinect_keypoints[mapping])
    
    return np.array(smpl_keypoints)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")
    
    frame = cv2.imread(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize\frame00000045.png")
    azure_keypoints_sample = np.array([(1152, 612), (1178, 542), (1197, 479), (1205, 373), (1221, 390), (1287, 413), (1339, 545), (1268, 592), (1248, 597), (1234, 608), (1209, 590), (1188, 391), (1126, 412), (1064, 525), (1046, 510), (1023, 463), (1035, 462), (1039, 451), (1199, 624), (1138, 733), (1131, 947), (1109, 1067), (1112, 602), (982, 616), (855, 796), (771, 783), (1207, 332), (1158, 312), (1182, 292), (1243, 298), (1159, 295), (1174, 300)])
    
    # Camera calibration
    K = np.array([[917.34753418,   0.        , 954.6260376 ],
                  [   0.        , 917.26507568, 554.48309326],
                  [   0.        ,   0.        ,   1.        ]])
    focal_length_value = (K[0,0] + K[1,1]) / 2.0
    camera_center = np.array([960, 540])
    
    print(azure_keypoints_sample.shape)

    # reshape
    azure_keypoints_sample = azure_keypoints_sample.reshape(32,2)

    # MAP from Kinect to openpose sequence of joints
    xy = map_kinect_to_smpl(azure_keypoints_sample)
    conf        = (xy != 0).any(axis=1, keepdims=True).astype(np.float32)
    smpl_kpts   = np.hstack([xy, conf])    # shape (49, 3)
    
    kpts        = smpl_kpts[None].astype(np.float32)    # (1, 49, 3)
    keypoints   = torch.from_numpy(kpts).to(device)
    print(f"keypoints shape = {azure_keypoints_sample.shape}")

    # Compute bounding box
    non_zero_mask = ~(azure_keypoints_sample == 0).any(axis=1)
    valid_xy      = azure_keypoints_sample[non_zero_mask]

    img_w=1920
    img_h=1080

    # ----- compute min / max per axis ----------------------------------
    x_min, y_min = valid_xy.min(axis=0)
    x_max, y_max = valid_xy.max(axis=0)

    pad = 150
    x_min, y_min = x_min - pad, y_min - pad
    x_max, y_max = x_max + pad, y_max + pad

    # clamp to image frame
    x_min = np.clip(x_min, 0, img_w-1)
    y_min = np.clip(y_min, 0, img_h - 1)
    x_max = np.clip(x_max, 0, img_w - 1)
    y_max = np.clip(y_max, 0, img_h - 1)

    bbox = (int(x_min), int(y_min), int(x_max), int(y_max))  # (x1, y1, x2, y2)
    #print(f"Bounding‑box (pad {pad}px):", bbox)

    w, h = x_max - x_min, y_max - y_min   

    
    # Load the pretrained CLIFF model.
    cliff = eval("cliff_hr48")
    cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

    norm_img = preprocess_frame(frame, target_size=(224, 224)).to(device)
    focal_length = torch.tensor([focal_length_value], dtype=torch.float32, device=device) #500
    camera_center_tensor = torch.tensor([camera_center], dtype=torch.float32, device=device)

    # Bounding box operations
    center = torch.tensor([[(x_min + x_max)/2.0,
                        (y_min + y_max)/2.0]], device=device)

    scale  = torch.tensor([max(w, h) / 200.0], device=device)
    b      = scale * 200.0

    
    # 1) stack into a (1×3) tensor [dx,dy,b]
    bbox_info = torch.stack([
        center[:, 0] - img_w / 2.0,   # dx
        center[:, 1] - img_h / 2.0,   # dy
        b                              # baseline
    ], dim=-1)                         # shape [1,3]

    # 2) normalize x,y by focal_length and multiply by 2.8
    #    (focal_length should be shape [1] or broadcastable)
    bbox_info[:, :2] = bbox_info[:, :2] \
                       / focal_length.unsqueeze(-1) \
                       * 2.8

    # 3) normalize the baseline entry
    bbox_info[:, 2] = (
        bbox_info[:, 2]
      - 0.24 * focal_length
    ) / (0.06 * focal_length)
    bbox_info = bbox_info.float() 
    
    # Run CLIFF
    with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = cliff_model(norm_img, bbox_info)
    
    img_h_t = torch.tensor([img_h], dtype=torch.float32, device=device)
    img_w_t = torch.tensor([img_w], dtype=torch.float32, device=device)

    # now both are tensors, so stack will work
    full_img_shape = torch.stack((img_h_t, img_w_t), dim=-1)   
    pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
    init_pose = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)        
    
    # Run SMPLify optimization
    smplify = SMPLify(step_size=1e-2, batch_size=1, num_iters=100, focal_length=focal_length)
    results = smplify(init_pose.detach(), pred_betas.detach(), pred_cam_full.detach(), camera_center_tensor, keypoints)
    
    new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
    
    # Get SMPL faces and vertices
    with torch.no_grad():
        pred_output = smpl(betas=new_opt_betas,
                           body_pose=new_opt_pose[:, 3:],global_orient=new_opt_pose[:, :3],
                           pose2rot=True,
                           transl=new_opt_cam_t)

    #vertices = pred_output.vertices.cpu().detach().numpy()
    vertices = pred_output.vertices.cpu().detach().numpy()
    if vertices.ndim == 3:
        vertices = vertices[0]
    faces = smpl.faces
    if not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(pyr_mesh)
    
    # Open pyrender window
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    main()
