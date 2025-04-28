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

# Joint map (for reference)
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

def preprocess_frame(frame, target_size=(224, 224)):
    """
    Preprocess the input frame (BGR) to a normalized tensor.
    Returns norm_img as a tensor of shape (1, 3, H, W) in float32.
    """
    resized = cv2.resize(frame, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
    norm_img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    return norm_img

def create_arrow(start, end, shaft_radius=0.005, head_radius=0.01, head_length=0.02, sections=20):
    """
    Create an arrow mesh from a start point (tail) to an end point (head) using trimesh.
    The arrow is built from a cylinder (shaft) and a cone (head).
    """
    vec = end - start
    total_length = np.linalg.norm(vec)
    if total_length < 1e-6:
        return None
    direction = vec / total_length

    # Reserve space for the arrow head.
    shaft_length = max(total_length - head_length, total_length * 0.8)
    head_length = total_length - shaft_length

    # Create the shaft as a cylinder along the Z-axis.
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length, sections=sections)
    shaft.apply_translation([0, 0, shaft_length / 2.0])

    # Create the head as a cone along the Z-axis.
    head = trimesh.creation.cone(radius=head_radius, height=head_length, sections=sections)
    head.apply_translation([0, 0, shaft_length + head_length / 2.0])

    # Combine shaft and head.
    arrow = trimesh.util.concatenate([shaft, head])

    # Align the arrow (default along Z) with the desired direction.
    z_axis = np.array([0, 0, 1])
    rot_matrix = trimesh.geometry.align_vectors(z_axis, direction)
    if rot_matrix is None:
        rot_matrix = np.eye(3)
    elif rot_matrix.shape == (4, 4):
        rot_matrix = rot_matrix[:3, :3]
    
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_matrix
    arrow.apply_transform(T_rot)

    # Translate so that its base (tail) is at the start position.
    arrow.apply_translation(start)
    return arrow

# Define state configurations.
STATE_CONFIG = {
    1: {
        'text': "Check knees visually for redness",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP LKnee")], joints[JOINT_NAMES.index("OP RKnee")]]  # Left and Right knees
    },
    2: {
        'text': "Check for redness and swelling",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LKnee")] + joints[JOINT_NAMES.index("OP LAnkle")]) / 2.0,  # Left calf (midpoint between knee and ankle)
            (joints[JOINT_NAMES.index("OP RKnee")] + joints[JOINT_NAMES.index("OP RAnkle")]) / 2.0   # Right calf (midpoint between knee and ankle)
        ]
    },
    3: {
        'text': "Check between toes",
        'targets': lambda joints: [
            (joints[JOINT_NAMES.index("OP LSmallToe")] + joints[JOINT_NAMES.index("OP LBigToe")]) / 2.0,  # Left toe (midpoint between big and small toe)
            (joints[JOINT_NAMES.index("OP RSmallToe")] + joints[JOINT_NAMES.index("OP RBigToe")]) / 2.0   # Right toe (midpoint between big and small toe)
        ]
    },
    4: {
        'text': "Check for swelling",
        'targets': lambda joints: [joints[JOINT_NAMES.index("OP RHeel")], joints[JOINT_NAMES.index("OP LHeel")]]  # Left and Right heels
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openpose', type=str, default=None,
                        help='Path to .json containing openpose detections (if available)')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # Load the pretrained CLIFF model.
    cliff = eval("cliff_hr48")
    cliff_model = cliff('./data/smpl_mean_params.npz').to(device)
    state_dict = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    state_dict = strip_prefix_if_present(state_dict, prefix="module.")
    cliff_model.load_state_dict(state_dict, strict=True)
    cliff_model.eval()

    # Load the SMPL model.
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

    # Set up the webcam and offscreen pyrender renderer.
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

    # Start with state 1.
    current_state = 1
    ret, frame = cap.read()
    img_h, img_w = frame.shape[:2]
    center = torch.tensor([[img_w / 2.0, img_h / 2.0]], device=device, dtype=torch.float32)
    scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    b = scale * 200
    bbox_info = torch.stack([center[:, 0] - img_w / 2.0,
                              center[:, 1] - img_h / 2.0, b], dim=-1)
    bbox_info[:, :2] = bbox_info[:, :2] / focal_length.unsqueeze(-1) * 2.8
    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * focal_length) / (0.06 * focal_length)

    full_img_shape = torch.stack((
            torch.tensor([img_h], device=device, dtype=torch.float32),
            torch.tensor([img_w], device=device, dtype=torch.float32)
        ), dim=-1)

    print("Starting real-time inference. Press 'q' to exit. Press 1/2/3/4 to change state.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_h, img_w = frame.shape[:2]
        norm_img = preprocess_frame(frame, target_size=(224, 224)).to(device)

        # Run the CLIFF model.
        with torch.no_grad():
            pred_rotmat, pred_betas, _ = cliff_model(norm_img, bbox_info)
        
        smpl_poses = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)
        '''
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, focal_length)
        
        
        camera_center = torch.hstack((
            torch.tensor([[img_w / 2.0]], device=device, dtype=torch.float32),
            torch.tensor([[img_h / 2.0]], device=device, dtype=torch.float32)
        )) / 2
        '''
        with torch.no_grad():
            pred_output = smpl(betas=pred_betas,
                               body_pose=smpl_poses[:, 3:],
                               global_orient=smpl_poses[:, :3],
                               pose2rot=True)
                               #transl=pred_cam_full)
        
        # Prepare the SMPL mesh.
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

        
        R_flip = tf.rotation_matrix(2*np.pi, [1, 1, 0])
        mesh.apply_transform(R_flip)
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)

        # Handle key presses for camera control and state change.
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
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                current_state = int(chr(key))
                print(f"State changed to {current_state}")

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

        # Recompute and add new arrows per frame.
        if hasattr(pred_output, 'joints'):
            joints = pred_output.joints.cpu().detach().numpy()[0]  # shape (num_joints, 3)
            # Subtract the same centroid to center the joints.
            joints_centered = joints - mesh_centroid
            if current_state in STATE_CONFIG:
                targets = STATE_CONFIG[current_state]['targets'](joints_centered)
                for target in targets:
                    # Define a constant tail offset. Adjust this as necessary.
                    offset = np.array([0.0, -0.02, -0.4])
                    head_offset = np.array([0.0, 0.0, -0.1])
                    target = target +head_offset
                    tail = target + offset

                    arrow_mesh = create_arrow(tail, target,  shaft_radius=0.01,head_radius=0.03, head_length=0.08)
                    if arrow_mesh is not None:
                        arrow_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1.0, 0.0, 0.0, 1.0))
                        arrow_mesh.apply_transform(R_flip)
                        arrow_pyrender = pyrender.Mesh.from_trimesh(arrow_mesh, material=arrow_material, smooth=False)

                        scene.add(arrow_pyrender)

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
        cv2.imshow("Webcam", frame)

    cap.release()
    offscreen_renderer.delete()
    cv2.destroyAllWindows()
