import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pytorch3d import transforms

from models.smpl import SMPL
from common import constants

from losses import *
from smplify import SMPLify

from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from common.utils import strip_prefix_if_present, cam_crop2full
# from common.mocap_dataset import MocapDataset  # not used in this live demo

# For 3D visualization
import pyrender
import trimesh

import trimesh.transformations as tf


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    '''
    # Load the pretrained CLIFF model.


    # Load the SMPL model.
    '''

    # Set up the cam and offscreen pyrender renderer.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    viewport_width, viewport_height = 1280, 720
    offscreen_renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)

    # Dummy parameters.
    dummy_detection = np.array([[0, 17, 37, 117, 142, 0.99, 0.99, 0]], dtype=np.float32)
    dummy_keypoints = np.zeros((1, 49, 3), dtype=np.float32)

    focal_length = torch.tensor([500.0], device=device, dtype=torch.float32)
    
    # Initialize camera control parameters.
    zoom_factor = 1.0
    angle_x = 0.0
    angle_y = 0.0
    rotate_step = np.radians(15)
    zoom_step = 0.1

    ## SMPL fitting and display
    while True:
        # Read in the current frame data
        RGB, keypoints = RGB_input, keypoints  ## Sample input


        '''
        The SMPL fitting components go here.

        '''
        # After you get the vertices and faces

        vertices = pred_output.vertices.cpu().detach().numpy()
        if vertices.ndim == 3:
            vertices = vertices[0]
        faces = smpl.faces
        if not isinstance(faces, np.ndarray):
            faces = faces.cpu().numpy() if torch.is_tensor(faces) else faces

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # Store the centroid then recenter the vertices.
        mesh_centroid = mesh.bounding_box.centroid.copy()
        mesh.vertices -= mesh_centroid

        # Optionally rotate the mesh if necessary
        #R_flip = tf.rotation_matrix(2*np.pi, [1, 1, 0])
        #mesh.apply_transform(R_flip)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)

        key = cv2.waitKey(1)
        if key != -1:
            if key == ord('q'):
                break
            elif key == ord('w'):
                angle_x -= rotate_step
            elif key == ord('s'):
                angle_x += rotate_step
            elif key == ord('a'):
                angle_y -= rotate_step
            elif key == ord('d'):
                angle_y += rotate_step
            elif key in [ord('+'), ord('=')]:
                zoom_factor *= (1 - zoom_step)
            elif key in [ord('-'), ord('_')]:
                zoom_factor *= (1 + zoom_step)

        # Compute camera pose.
        mesh_extent = np.max(mesh.bounding_box.extents)
        base_distance = mesh_extent * 2.5
        camera_distance = base_distance * zoom_factor

        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x), 0],
            [0, np.sin(angle_x), np.cos(angle_x), 0],
            [0, 0, 0, 1]
        ])
        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
            [0, 0, 0, 1]
        ])
        T = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, camera_distance],
            [0, 0, 0, 1]
        ])
        cam_pose = R_y @ R_x @ T

        # Build the pyrender scene.
        scene = pyrender.Scene()
        scene.add(mesh_pyrender)


        camera_obj = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera_obj, pose=cam_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene.add(light, pose=cam_pose)

        color, depth = offscreen_renderer.render(scene)
        # Make a writable copy for overlaying text.
        color_writable = color.copy()
        if current_state in STATE_CONFIG:
            state_text = STATE_CONFIG[current_state]['text']
            text_pos = (viewport_width - 500, 30)
            cv2.putText(color_writable, state_text, text_pos,
                        cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("3D SMPL", color_writable[:, :, ::-1])
        cv2.imshow("RGB frame", RGB)

    cap.release()
    offscreen_renderer.delete()
    cv2.destroyAllWindows()
