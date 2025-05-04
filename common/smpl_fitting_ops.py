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

from common.depth_operations import px_to_cam, get_pelvis_translation, get_probe_centroid, depth_scaled_metric
from common.scene_operations import rgb_hsv_mask, centroid_px, add_reference_frame, add_prob_centroid2scene

from losses import camera_fitting_loss, body_fitting_loss

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from prior import MaxMixturePrior

def refine_smpl(smpl,betas, init_pose, pelvis_translation,pred_cam_full, keypoints, camera_center_tensor, focal_length,pose_prior, num_iters=5, device="cuda"):
    # Make camera translation learnable parameter
    #camera_translation = pred_cam_full.clone()
    JOINT_NAMES = [ 'OP Neck',                                
                'OP LSmallToe',
                'OP LHeel',
                'OP RHip', 'OP LHip', 'Right Hip', 'Left Hip',
                'OP RSmallToe',
                'OP RHeel',
                'Neck (LSP)',
                'Top of Head (LSP)',
                'Pelvis (MPII)',
                'Spine (H36M)',
                'Jaw (H36M)',
                'Head (H36M)',
                ]
    
    ign_joints = [constants.JOINT_IDS[i] for i in JOINT_NAMES]

    # Get joint confidence
    joints_2d = keypoints[:, :, :2]
    joints_conf = keypoints[:, :, -1]
    
    # Split SMPL pose to body pose and global orientation
    body_pose = init_pose[:, 3:]
    global_orient = init_pose[:, :3]

    pred_cam_full.requires_grad = False

    # Optimize only the body pose and global orientation of the body
    body_pose.requires_grad=True
    betas.requires_grad=False
    global_orient.requires_grad=True
    pred_cam_full.requires_grad = False
    body_opt_params = [body_pose, betas, global_orient]

    
    keypoints[:, ign_joints] = 0
    body_optimizer = torch.optim.AdamW(body_opt_params, lr= 1e-2, betas=(0.9, 0.999))

    assert body_pose.device.type== 'cuda'
    assert betas.device.type =='cuda'
    assert pred_cam_full.device.type  == 'cuda'
    assert global_orient.device.type  == 'cuda'


    
    for i in range(num_iters):
        start_time = time.time()
        smpl_output = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, pose2rot=True, transl=pelvis_translation)
        assert smpl_output.joints.device.type == 'cuda'
        loss = body_fitting_loss(body_pose, betas, smpl_output.joints, pred_cam_full, camera_center_tensor,
                                 keypoints[:, :, :2], keypoints[:, :, -1], pose_prior,
                                 focal_length=focal_length)
        body_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        body_optimizer.step()
        end_time = time.time()

    with torch.no_grad():
            smpl_output = smpl(betas=betas,
                               body_pose=body_pose,
                               global_orient=global_orient,
                               pose2rot=True,
                               transl=pelvis_translation)
    new_opt_vertices = smpl_output.vertices.detach()
    new_opt_joints = smpl_output.joints.detach()
    new_opt_pose = torch.cat([global_orient, body_pose], dim=-1).detach()
    new_opt_betas = betas.detach()
    faces = smpl.faces

    return new_opt_vertices, new_opt_joints, new_opt_pose,new_opt_betas, faces

def smpl_skip_refinement(smpl, betas, init_pose, pose2rot, transl):
    with torch.no_grad():
        pred_output = smpl(betas=betas,
                       body_pose=init_pose[:, 3:],
                       global_orient=init_pose[:, :3],
                       pose2rot=True,
                       transl=transl)    
    return pred_output.vertices, smpl.faces