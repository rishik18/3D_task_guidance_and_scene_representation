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

def rgb_hsv_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[..., 0] / 179.0          # OpenCV hue range → 0-1
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    mask = ((h >= 0.044) & (h <= 0.170) &
            (s >= 0.437) & (s <= 0.893) &
            (v >= 0.820) & (v <= 1.000)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255
    return mask

def centroid_px(mask):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return None
    return int(round(M["m10"]/M["m00"])), int(round(M["m01"]/M["m00"]))

def add_reference_frame(scene):
    axis_trimesh = trimesh.creation.axis(
        origin_size = 0.03,     # little cube at the origin
        axis_length = 0.3)      # length of each arrow (metres)

    axis_mesh   = pyrender.Mesh.from_trimesh(axis_trimesh, smooth=False)
    axis_node   = scene.add(axis_mesh, pose=np.eye(4))
    return

def add_prob_centroid2scene(scene, centroid_3d, size):
    # Add Probe centroid as a shpere
    # ---------- add a red sphere at the centroid (if we have one) -------------
    if centroid_3d is not None:
        sphere_trimesh = trimesh.creation.icosphere(subdivisions=3, radius=size)
        sphere_trimesh.visual.vertex_colors = [255, 0, 0, 255]   # RGBA red
        sphere = pyrender.Mesh.from_trimesh(sphere_trimesh, smooth=False)
        sphere_node = scene.add(sphere, pose=np.eye(4))          # identity pose
        scene.set_pose(sphere_node, np.block([
            [np.eye(3), centroid_3d.reshape(3,1)],
            [np.zeros((1,3)), 1]
        ]))
    return