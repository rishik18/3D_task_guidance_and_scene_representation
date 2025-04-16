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

    azure_keypoints_sample = np.array([[(401, 493), (388, 392), (376, 305), (355, 164), (383, 190), (476, 221), (531, 370), (539, 454), (523, 504), (505, 561)
                                        , (481, 505), (333, 186), (240, 203), (158, 371), (209, 454), (201, 495), (272, 548)
                                        , (281, 467), (460, 489), (543, 678), (577, 811), (613, 890), (344, 496), (365, 716), (400, 864), (419, 951), (349, 107), (373, 93)
                                        , (384, 65), (406, 71), (343, 62), (276, 51)]])
    print(azure_keypoints_sample.shape)
    azure_keypoints_sample = azure_keypoints_sample.reshape(32,2)
    print(azure_keypoints_sample.shape)
    print(azure_keypoints_sample)

    smpl_keypoints = map_kinect_to_smpl(azure_keypoints_sample)
    
    print("Mapped SMPL keypoints shape:", smpl_keypoints.shape)
    print("SMPL keypoints:")
    print(smpl_keypoints)

    # add column of ones for joint confidence
    confi_ones = np.ones((smpl_keypoints.shape[0], 1))
    smpl_keypoints = np.hstack((smpl_keypoints, confi_ones))
    print("Final shape after appending homogeneous coordinate:", smpl_keypoints.shape)
    print("Final keypoints array:")
    print(smpl_keypoints)


    # Convert 2D keypoints to torch tensor and add batch dimension: shape (1, 49, 3)
    keypoints = torch.from_numpy(smpl_keypoints[None]).float().to(device)
    
    # Define camera parameters for SMPLify.

    img_w, img_h = 1920.0, 1080.0
    
    # Initialize SMPL parameters: zero pose (72), zero shape (10), and dummy camera (3).
    init_pose = torch.zeros((1, 72), dtype=torch.float32, device=device)
    init_betas = torch.zeros((1, 10), dtype=torch.float32, device=device)
    #init_cam = torch.zeros((1, 3), dtype=torch.float32, device=device)

    # Camera calibration
    K = np.array([[917.34753418,   0.        , 954.6260376 ],
                  [   0.        , 917.26507568, 554.48309326],
                  [   0.        ,   0.        ,   1.        ]])
    focal_length_value = (K[0,0] + K[1,1]) / 2.0
    camera_center = np.array([K[0,2], K[1,2]])
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    focal_length = torch.tensor([focal_length_value], dtype=torch.float32, device=device)
    camera_center_tensor = torch.tensor([camera_center], dtype=torch.float32, device=device)
    
    translation = np.array([-32.05845642, -2.09264588, 3.95474815])
    R = np.array([[ 9.99998212e-01,  1.68047613e-03, -8.45391944e-04],
                  [-1.59707933e-03,  9.95895743e-01,  9.04935747e-02],
                  [ 9.93994530e-04, -9.04920623e-02,  9.95896697e-01]])
    
    # Initialize the camera translation using the calibration:
    init_cam = torch.tensor([translation], dtype=torch.float32, device=device)
    
    # Load the SMPL model (ensure your SMPL model directory is set correctly in constants)
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)
    
    # Run SMPLify optimization
    smplify = SMPLify(step_size=1e-2, batch_size=1, num_iters=100, focal_length=focal_length)
    results = smplify(init_pose.detach(), init_betas.detach(), init_cam.detach(), camera_center_tensor, keypoints)
    
    new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, new_opt_joint_loss = results
    
    with torch.no_grad():
        pred_output = smpl(betas=new_opt_betas,
                           body_pose=new_opt_pose[:, 3:],global_orient=new_opt_pose[:, :3],
                           pose2rot=True,
                           transl=init_cam)
        vertices = pred_output.vertices.cpu().numpy().squeeze()
    
    # Render the SMPL mesh with pyrender on a blank background.
    mesh = trimesh.Trimesh(vertices, smpl.faces)
    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(pyr_mesh)
    
    # Open pyrender window
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    main()
