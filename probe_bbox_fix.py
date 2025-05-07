
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
from common.utils import strip_prefix_if_present, cam_crop2full, full2crop_cam
import time
from common.imutils import process_image 

from common.depth_operations import px_to_cam, get_pelvis_translation, get_probe_centroid, depth_scaled_metric, smpl_fix_coordinates,project_cam_to_px
from common.scene_operations import rgb_hsv_mask, centroid_px, add_reference_frame, add_prob_centroid2scene, add_custom_marker_az
from common.smpl_fitting_ops import refine_smpl, smpl_skip_refinement
from common.preprocessing_operations import map_kinect_to_smpl, process_keypoints, get_crop_cam, preprocess_crop, compute_bbox_full_scale
from losses import camera_fitting_loss, body_fitting_loss
from common.renderer_pyrd import Renderer

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from prior import MaxMixturePrior

# For the GMM prior, we use the GMM implementation of SMPLify-X
# https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py
from prior import MaxMixturePrior


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device}")
    
    #frame = cv2.imread(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize\frame00000083.png")
    #azure_keypoints_sample = np.array([(1163, 625), (1167, 549), (1172, 487), (1167, 384), (1185, 399), (1256, 409), (1288, 542), (1219, 599), (1178, 590), (1131, 599), (1143, 584), (1151, 403), (1091, 424), (1060, 553), (1109, 607), (1143, 569), (1194, 585), (1139, 587), (1213, 631), (1329, 743), (1365, 931), (1390, 1032), (1119, 620), (986, 689), (856, 879), (751, 898), (1164, 344), (1123, 325), (1145, 305), (1201, 310), (1121, 308), (1129, 312)])
    #depth_path = r"D:\CSU_data\mockup_video_data_v2\tmp_depth_177720_resize\frame00000083.png"

    #frame = cv2.imread(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize\frame00000045.png")
    #azure_keypoints_sample = np.array([(1152, 612), (1178, 542), (1197, 479), (1205, 373), (1221, 390), (1287, 413), (1339, 545), (1268, 592), (1248, 597), (1234, 608), (1209, 590), (1188, 391), (1126, 412), (1064, 525), (1046, 510), (1023, 463), (1035, 462), (1039, 451), (1199, 624), (1138, 733), (1131, 947), (1109, 1067), (1112, 602), (982, 616), (855, 796), (771, 783), (1207, 332), (1158, 312), (1182, 292), (1243, 298), (1159, 295), (1174, 300)])
    #depth_path = r"D:\CSU_data\mockup_video_data_v2\tmp_depth_177720_resize\frame00000045.png"

    frame = cv2.imread(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize\frame00000100.png")
    azure_keypoints_sample = np.array([(1165, 624), (1168, 548), (1171, 487), (1166, 383), (1183, 398), (1254, 411), (1290, 543), (1219, 598), (1176, 590), (1130, 600), (1143, 584), (1150, 402), (1090, 424), (1058, 553), (1114, 607), (1144, 567), (1175, 596), (1141, 582), (1215, 629), (1316, 749), (1367, 936), (1396, 1039), (1121, 620), (990, 688), (862, 877), (762, 894), (1163, 343), (1120, 327), (1142, 307), (1198, 309), (1118, 311), (1127, 314)])
    depth_path = r"D:\CSU_data\mockup_video_data_v2\tmp_depth_177720_resize\frame00000100.png"

    #frame = cv2.imread(r"D:\CSU_data\mockup_video_data_v2\tmp_color_199735_resize\frame00000101.png")
    #azure_keypoints_sample = np.array([(1164, 625), (1168, 549), (1171, 487), (1166, 384), (1183, 399), (1254, 410), (1290, 543), (1220, 597), (1177, 591), (1132, 600), (1143, 585), (1150, 403), (1090, 424), (1058, 553), (1114, 608), (1144, 568), (1168, 590), (1140, 580), (1215, 629), (1316, 752), (1367, 936), (1396, 1036), (1120, 620), (989, 689), (858, 878), (757, 894), (1163, 344), (1121, 326), (1142, 306), (1198, 309), (1118, 310), (1126, 314)])
    #depth_path = r"D:\CSU_data\mockup_video_data_v2\tmp_depth_177720_resize\frame00000101.png"
    
    # Camera calibration
    K = np.array([[917.34753418,   0.        , 954.6260376 ],
                  [   0.        , 917.26507568, 554.48309326],
                  [   0.        ,   0.        ,   1.        ]])
    focal_length_value = (K[0,0] + K[1,1]) / 2.0
    camera_center = [954.6260376, 554.48309326]

    ## Depth map input scaled from 0-255
    depth_u8 = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    ## Scale the depth to metric based on Kinect documentation
    depth_m = depth_scaled_metric(depth_u8, near = 0.5, far = 5.5, offset = 0.3)
    
    ## process keypoints
    keypoints = process_keypoints(azure_keypoints_sample, device)
    
    ## Image height and width
    img_w=1920
    img_h=1080

    # Get translation for SMPL
    pelvis_translation = get_pelvis_translation(azure_keypoints_sample[0],depth_m, K, device)
    # Compute bounding box

    
    bbox, w, h = compute_bbox_full_scale(azure_keypoints_sample, img_w, img_h)
    
    # Load the pretrained CLIFF model.
    cliff = eval("cliff_hr48")
    cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Load SMPL model
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

    # Load pose prior
    pose_prior = MaxMixturePrior(prior_folder='data',num_gaussians=8,dtype=torch.float32).to(device)

    #Image shape tensor
    img_h_t = torch.tensor([img_h], dtype=torch.float32, device=device)
    img_w_t = torch.tensor([img_w], dtype=torch.float32, device=device)
    full_img_shape = torch.stack((img_h_t, img_w_t), dim=-1) 

    # Focal len and cam center tensors
    focal_length = torch.tensor([focal_length_value], dtype=torch.float32, device=device) #500
    camera_center_tensor = torch.tensor([camera_center], dtype=torch.float32, device=device)

    # Get 3D coordinates of probe centroid
    centroid_3d = get_probe_centroid(frame, depth_m, K)
    
    # Get proprocessing data for CLIFF 
    # For debuggin visualize crop_img to check the image fed to CLIFF
    norm_img, center, scale, ul, br, crop_img, bbox_info, b = preprocess_crop(frame, bbox, crop_height =224, crop_width=224,
                                                                              camera_center=camera_center
                                                                              , focal_length=focal_length, device=device)



    ### Pass the Kinect translation to get_crop_cam to get the camera initialization
    kinect_translation = [-32.05845642,  -2.09264588,   3.95474815]
    init_cam = get_crop_cam(kinect_translation,center, b, camera_center_tensor, focal_length, device =device)

    # Run CLIFF
    with torch.no_grad():
            pred_rotmat, betas, pred_cam_crop = cliff_model(norm_img, bbox_info, n_iter=5 )

    # Use pred_cam_full if using CLIFF predicted cam
    #pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)

    # Use data_full_cam if using translation data from Kinect
    data_full_cam = torch.tensor([[kinect_translation[0]/1000,  -kinect_translation[1]/1000,   kinect_translation[2]/1000]], dtype=torch.float32, device=device)
    
    # Process pose data
    init_pose = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)    
    
    # For reference - >refine_smpl(smpl,betas, init_pose, pelvis_translation,pred_cam_full, keypoints, camera_center_tensor, focal_length,pose_prior, num_iters=5, device="cuda")
    new_opt_vertices, new_opt_joints, new_opt_pose,new_opt_betas, faces = refine_smpl(smpl = smpl, betas=betas,
                                                                                      init_pose= init_pose, 
                                                                                      pelvis_translation =pelvis_translation,
                                                                                      pred_cam_full =data_full_cam,
                                                                                      keypoints= keypoints,
                                                                                      camera_center_tensor =camera_center_tensor,
                                                                                      focal_length = focal_length,
                                                                                      pose_prior =pose_prior,
                                                                                      num_iters=5,
                                                                                      device= device)
    
    
    ##smpl_skip_refinement sample
    #new_opt_vertices, faces = smpl_skip_refinement(smpl = smpl, betas =betas, init_pose=init_pose, pose2rot=True, transl=pelvis_translation)
    
    # adjusts SMPL pelvis to origin before applying the pelvis translation
    new_opt_vertices, new_joints = smpl_fix_coordinates(new_opt_vertices,new_opt_joints, pelvis_translation)

    ## Rendering section
    vertices = new_opt_vertices.cpu().detach().numpy()
    new_joints = new_joints.cpu().detach().numpy()

    if vertices.ndim == 3:
        vertices = vertices[0]
    if not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    

    scene = pyrender.Scene()

    # refer to JOINT_NAMES in the common/preprocessing_operations for joint numbers

    # Add a marker on the right leg
    add_custom_marker_az(scene,
                         mesh,
                         new_joints,
                         joint_id_a=10,            # 'OP RWrist'
                         joint_id_b=11,            # 'OP LWrist'
                         l=0.3,                  # 10 cm from RWrist toward LWrist
                         alpha_deg=-120.0,       # azimuth inside X-Z plane
                         add_shpere = True,
                         sphere_radius=0.02,
                         sphere_color=(0.2,0.8,1.0,1.0)
                         ,add_color_gradient = True
                         ,inner_r  = 0.005
                         ,outer_r = 0.035
                         ,hit_rgba = np.array([255, 0, 0, 255], np.uint8)
                         ,gamma = 0.2)
    # Add another marker on the right leg at differet length
    add_custom_marker_az(scene,
                         mesh,
                         new_joints,
                         joint_id_a=10,            # 'OP RWrist'
                         joint_id_b=11,            # 'OP LWrist'
                         l=0.15,                  # 10 cm from RWrist toward LWrist
                         alpha_deg=-90.0,       # azimuth inside X-Z plane
                         add_shpere = False,
                         sphere_radius=0.02,
                         sphere_color=(0.2,0.8,1.0,1.0)
                         ,add_color_gradient = True
                         ,inner_r  = 0.005
                         ,outer_r = 0.02
                         ,hit_rgba = np.array([0, 255, 0, 255], np.uint8)
                         ,gamma = 0.2)


    # Add a marker on the left leg with changed color and angle
    add_custom_marker_az(scene,
                         mesh,
                         new_joints,
                         joint_id_a=13,            # 'OP RWrist'
                         joint_id_b=14,            # 'OP LWrist'
                         l=0.2,                  # 10 cm from RWrist toward LWrist
                         alpha_deg=-50.0,       # azimuth inside X-Z plane
                         add_shpere = True,
                         sphere_radius=0.02,
                         sphere_color=(0.0,0.0,1.0,1.0)
                         ,add_color_gradient = True
                         ,inner_r  = 0.005
                         ,outer_r = 0.035
                         ,hit_rgba = np.array([0, 255, 0, 255], np.uint8)
                         ,gamma = 0.2)

    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(pyr_mesh)

    # Use this line to add reference frame to scene
    add_reference_frame(scene)

    # Use this line to add probe centroid marker to scene
    add_prob_centroid2scene(scene, centroid_3d = centroid_3d, size = 0.04)

    # Open pyrender window
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    main()
