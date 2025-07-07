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
import math


def rgb_hsv_mask(bgr, min_area= 400):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[..., 0] / 179.0
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0

    # initial threshold
    mask = ((0.044 <= h) & (h <= 0.170) &
            (0.437 <= s) & (s <= 0.893) &
            (0.820 <= v) & (v <= 1.0)).astype(np.uint8) * 255

    # clean–up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # connected components
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if n > 1:
        # index of biggest component among stats[1:] (skip background)
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_area = stats[largest_idx, cv2.CC_STAT_AREA]

        if largest_area >= min_area:
            mask = (labels == largest_idx).astype(np.uint8) * 255
        else:
            # too small → return empty mask
            mask = np.zeros_like(mask, dtype=np.uint8)
    else:
        # no foreground blobs detected
        mask = np.zeros_like(mask, dtype=np.uint8)

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

def add_custom_marker_az(scene,
                         mesh_trimesh,          # body mesh (trimesh.Trimesh)
                         joints,                # (N,3) numpy
                         joint_id_a,
                         joint_id_b,
                         l,               # % length from A along AB
                         alpha_deg,       # azimuth inside X-Z plane
                         add_shpere = False,
                         sphere_radius=0.02,
                         sphere_color=(0.2,0.8,1.0,1.0)
                         ,add_color_gradient = True
                         ,inner_r  = 0.01
                         ,outer_r = 0.03
                         ,hit_rgba = np.array([255, 64, 32, 255], np.uint8)
                         ,gamma = 2.5):
    """
    l is percentage of length along the line from joint A to joint B
    α is measured in the plane perpendicular to AB:
        0° → local +X
       90° → local +Z
      180° → local −X      (-y is impossible because the whole plane is ⟂ y)
      gamma  # >1 steeper, <1 gentler, 1 = linear gamma controls the gradient

     # refer to JOINT_NAMES in the common/preprocessing_operations for joint numbers
      
     THis stuff controls the sphere addition, radius and color
     add_shpere = False,
     sphere_radius=0.02,
     sphere_color=(0.2,0.8,1.0,1.0)
        
      The below stuff controls the color gradient and its color
        add_color_gradient = True
      ,inner_r  = 0.01
      ,outer_r = 0.03
      ,hit_rgba = np.array([255, 64, 32, 255], np.uint8)
      ,gamma = 2.5)
    Returns hit-point or None.
    """
    pA, pB = joints[0,joint_id_a], joints[0,joint_id_b]
    vAB    = pB - pA
    normAB = np.linalg.norm(vAB)
    if normAB < 1e-8:
        raise ValueError("AB length ≈ 0; cannot build frame.")

    # local frame ---------------------------------------------------
    y_axis = vAB / normAB                                            # +Y
    # choose an arbitrary world-up that is *not* colinear with y_axis
    world_up = np.array([0, 0, 1.0])
    if abs(np.dot(world_up, y_axis)) > 0.95:                         # almost colinear
        world_up = np.array([0, 1.0, 0])
    x_axis = np.cross(world_up, y_axis)
    x_axis /= np.linalg.norm(x_axis)                                 # +X
    z_axis = np.cross(y_axis, x_axis)                                # +Z (right-handed)

    # ray origin & direction ---------------------------------------
    origin   = pA + l * vAB                               # on AB
    alpha    = math.radians(alpha_deg)
    dir_vec  =  math.cos(alpha) * x_axis + math.sin(alpha) * z_axis
    dir_vec /= np.linalg.norm(dir_vec)

    # ray-mesh intersection ----------------------------------------
    loc, *_ = mesh_trimesh.ray.intersects_location(
                  origin.reshape(1,3), dir_vec.reshape(1,3),
                  multiple_hits=False)
    if len(loc)==0:
        return None
    hit = loc[0]

        # ───────────────── colour-gradient around the hit ─────────────────
    if add_color_gradient:
                
        dists   = np.linalg.norm(mesh_trimesh.vertices - hit, axis=1)
        in_band = dists < outer_r
        
        if np.any(in_band):
                            
            lin   = (outer_r - dists[in_band]) / (outer_r - inner_r)
            w     = np.clip(lin, 0.0, 1.0) ** gamma
            w     = w[:, None]                                  # (k,1)
        
            base  = mesh_trimesh.visual.vertex_colors[in_band].astype(np.float32)
            target = hit_rgba.astype(np.float32)                # make it float for math
            blend = (w * target + (1.0 - w) * base).astype(np.uint8)
        
            mesh_trimesh.visual.vertex_colors[in_band] = blend

    # visual sphere -------------------------------------------------
    if add_shpere:
        sph = trimesh.creation.uv_sphere(radius=sphere_radius)
        sph.visual.vertex_colors = sphere_color
        sph.apply_translation(hit)
        scene.add(pyrender.Mesh.from_trimesh(sph))
    return hit