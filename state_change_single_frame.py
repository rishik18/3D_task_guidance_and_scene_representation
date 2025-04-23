# -*- coding: utf-8 -*-
"""
Interactive single‑frame indicator viewer

* Press **1 – 4** to cycle through the four clinical‑check states.
* Red arrows and on‑screen text update in real time.
* No external fonts required – caption uses the built‑in bitmap font.

Tested with:
  • Python 3.10  • PyTorch 2.2  • PyRender 0.1.45  • Pyglet 2.0.10

Parts of this code have been edited or debugged using ChatGPT- O3 model
"""

import numpy as np
import torch
import cv2
import trimesh
import pyrender
from pyrender.constants import TextAlign
from pytorch3d import transforms

from common import constants
from common.utils import strip_prefix_if_present, cam_crop2full
from models.smpl import SMPL
from models.cliff_hr48.cliff import CLIFF as cliff_hr48
from smplify import SMPLify

# -----------------------------------------------------------------------------
# GLOBAL STATE ----------------------------------------------------------------
# -----------------------------------------------------------------------------
current_state: int        = 1      # 1‑based state index (1‑4)
arrow_nodes:   list       = []     # PyRender nodes that hold the arrows
joints_centered: np.ndarray | None = None   # will be filled after SMPL
mesh_centroid: np.ndarray | None   = None
R_flip = np.diag([1, -1, -1, 1]).astype(np.float32)  # camera→world flip

# -----------------------------------------------------------------------------
# Data‑driven configuration for each clinical check ---------------------------
# -----------------------------------------------------------------------------
JOINT_NAMES = [
    # 0‑24  OpenPose, 25‑48  ground‑truth superset
    'OP Nose','OP Neck','OP RShoulder','OP RElbow','OP RWrist','OP LShoulder',
    'OP LElbow','OP LWrist','OP MidHip','OP RHip','OP RKnee','OP RAnkle',
    'OP LHip','OP LKnee','OP LAnkle','OP REye','OP LEye','OP REar','OP LEar',
    'OP LBigToe','OP LSmallToe','OP LHeel','OP RBigToe','OP RSmallToe',
    'OP RHeel',
    # ground‑truth (we still keep the indices to 48 for completeness)
    'Right Ankle','Right Knee','Right Hip','Left Hip','Left Knee','Left Ankle',
    'Right Wrist','Right Elbow','Right Shoulder','Left Shoulder','Left Elbow',
    'Left Wrist','Neck (LSP)','Top of Head (LSP)','Pelvis (MPII)',
    'Thorax (MPII)','Spine (H36M)','Jaw (H36M)','Head (H36M)','Nose',
    'Left Eye','Right Eye','Left Ear','Right Ear'
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
STATE_CONFIG = {
    1: dict(
        text    = "Check knees visually for redness",
        targets = lambda j: [j[JOINT_NAMES.index('OP LKnee')],
                             j[JOINT_NAMES.index('OP RKnee')]]),

    2: dict(
        text    = "Check for redness and swelling (calves)",
        targets = lambda j: [(j[JOINT_NAMES.index('OP LKnee')] +
                              j[JOINT_NAMES.index('OP LAnkle')]) * 0.5,
                             (j[JOINT_NAMES.index('OP RKnee')] +
                              j[JOINT_NAMES.index('OP RAnkle')]) * 0.5]),

    3: dict(
        text    = "Check between toes",
        targets = lambda j: [(j[JOINT_NAMES.index('OP LSmallToe')] +
                              j[JOINT_NAMES.index('OP LBigToe')]) * 0.5,
                             (j[JOINT_NAMES.index('OP RSmallToe')] +
                              j[JOINT_NAMES.index('OP RBigToe')]) * 0.5]),

    4: dict(
        text    = "Check heels for swelling",
        targets = lambda j: [j[JOINT_NAMES.index('OP RHeel')],
                             j[JOINT_NAMES.index('OP LHeel')]]),
}

# -----------------------------------------------------------------------------
# Utility functions -----------------------------------------------------------
# -----------------------------------------------------------------------------

def create_arrow(start: np.ndarray, end: np.ndarray,
                 shaft_radius=0.01, head_radius=0.03, head_length=0.08):
    """Return a trimesh arrow from *start* → *end* (3‑D np arrays)."""
    vec   = end - start
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None

    direction   = vec / length
    shaft_len   = max(length - head_length, length * 0.8)
    head_length = length - shaft_len

    shaft = trimesh.creation.cylinder(radius=shaft_radius,
                                      height = shaft_len,
                                      sections=20)
    shaft.apply_translation([0, 0, shaft_len * 0.5])

    head  = trimesh.creation.cone(radius=head_radius,
                                  height=head_length,
                                  sections=20)
    head.apply_translation([0, 0, shaft_len + head_length * 0.5])

    arrow = trimesh.util.concatenate([shaft, head])

    z = np.array([0, 0, 1])
    R = trimesh.geometry.align_vectors(z, direction)
    if R is None:
        R = np.eye(3)
    elif R.shape == (4, 4):
        R = R[:3, :3]

    T = np.eye(4)
    T[:3, :3] = R
    arrow.apply_transform(T)
    arrow.apply_translation(start)
    return arrow

# -----------------------------------------------------------------------------
# Arrow drawing & key‑callback helpers ----------------------------------------
# -----------------------------------------------------------------------------

def _draw_arrows(scene: pyrender.Scene):
    """Delete old arrows and draw the new set for *current_state*."""
    global arrow_nodes

    # remove previous arrows
    for n in arrow_nodes:
        scene.remove_node(n)
    arrow_nodes.clear()

    if joints_centered is None or current_state not in STATE_CONFIG:
        return

    for tgt in STATE_CONFIG[current_state]['targets'](joints_centered):
        tail  = tgt + np.array([0.0, -0.02, -0.40])
        head  = tgt + np.array([0.0,  0.00, -0.10])
        arrow = create_arrow(tail, head)
        if arrow is None:
            continue
        #arrow.apply_transform(R_flip)
        mat   = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1, 0, 0, 1))
        node  = scene.add(pyrender.Mesh.from_trimesh(arrow, material=mat, smooth=False))
        arrow_nodes.append(node)


def _update_state(viewer: pyrender.Viewer, new_state: int):
    """Callback executed on key press."""
    global current_state
    current_state = new_state

    # update arrows & caption
    _draw_arrows(viewer.scene)
    viewer.viewer_flags['caption'][0]['text'] = STATE_CONFIG[new_state]['text']

# -----------------------------------------------------------------------------
# Main pipeline ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # ------------------------------------------------------------------
    # 1  Load colour frame & Azure‑Kinect 2‑D joints (example numbers) --
    # ------------------------------------------------------------------
    frame = cv2.imread(r"D:/CSU_data/mockup_video_data_v2/tmp_color_199735_resize/frame00000045.png")
    azure_2d = np.array([
        (1152, 612),(1178, 542),(1197, 479),(1205, 373),(1221, 390),
        (1287, 413),(1339, 545),(1268, 592),(1248, 597),(1234, 608),
        (1209, 590),(1188, 391),(1126, 412),(1064, 525),(1046, 510),
        (1023, 463),(1035, 462),(1039, 451),(1199, 624),(1138, 733),
        (1131, 947),(1109,1067),(1112, 602),( 982, 616),( 855, 796),
        ( 771, 783),(1207, 332),(1158, 312),(1182, 292),(1243, 298),
        (1159, 295),(1174, 300)
    ], dtype=np.float32).reshape(32, 2)

    # camera intrinsics ------------------------------------------------
    K_cam = np.array([[917.3475, 0.0, 954.6260],
                      [0.0, 917.2651, 554.4831],
                      [0.0, 0.0, 1.0]])
    focal_scalar = 0.5 * (K_cam[0, 0] + K_cam[1, 1])
    cam_centre   = np.array([960, 540], dtype=np.float32)

    # ------------------------------------------------------------------
    # 2  Convert Azure joints → OpenPose → SMPL order -------------------
    # ------------------------------------------------------------------

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

    smpl_xy   = map_kinect_to_smpl(azure_2d)
    conf      = (smpl_xy != 0).any(axis=1, keepdims=True).astype(np.float32)
    smpl_kpts = np.hstack([smpl_xy, conf])[None].astype(np.float32)  # (1,49,3)
    keypoints = torch.from_numpy(smpl_kpts).to(device)

    # ------------------------------------------------------------------
    # 3  Run CLIFF → SMPLify -------------------------------------------
    # ------------------------------------------------------------------
    norm_img = cv2.resize(frame, (224, 224)).astype(np.float32)
    norm_img = (norm_img / 255. - 0.5) / 0.5  # [-1,1]
    norm_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

    # CLIFF ------------------------------------------------------------
    cliff = cliff_hr48('./data/smpl_mean_params.npz').to(device)
    sd    = torch.load('./data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt')['model']
    cliff.load_state_dict(strip_prefix_if_present(sd, 'module.'), strict=True)
    cliff.eval()

    # bounding box (pad 150 px) ---------------------------------------
    nz   = azure_2d[(azure_2d != 0).any(1)]
    x0,y0 = nz.min(0) - 150
    x1,y1 = nz.max(0) + 150
    x0,y0 = np.clip([x0,y0], 0, [1919,1079])
    x1,y1 = np.clip([x1,y1], 0, [1919,1079])
    w,h   = x1-x0, y1-y0

    center = torch.tensor([[(x0+x1)/2, (y0+y1)/2]], device=device)
    scale  = torch.tensor([max(w, h) / 200.0], device=device)
    b      = scale * 200.0

    bbox_info = torch.stack([
        center[:,0] - 1920/2,
        center[:,1] - 1080/2,
        b], dim=-1)
    fl_t   = torch.tensor([focal_scalar], device=device)
    bbox_info[:,:2] = bbox_info[:,:2] / fl_t.unsqueeze(-1) * 2.8
    bbox_info[:, 2] = (bbox_info[:,2] - 0.24*fl_t) / (0.06*fl_t)
    bbox_info = bbox_info.float()

    with torch.no_grad():
        pred_R, pred_b, pred_cam_crop = cliff(norm_img, bbox_info)

    img_shape = torch.tensor([[1080,1920]], dtype=torch.float32, device=device)
    pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, img_shape, fl_t)
    init_pose = transforms.matrix_to_axis_angle(pred_R).view(-1,72)

    smplify = SMPLify(step_size=1e-2, batch_size=1, num_iters=100, focal_length=fl_t)
    smpl = SMPL(constants.SMPL_MODEL_DIR, batch_size=1).to(device)

    res = smplify(init_pose.detach(), pred_b.detach(), pred_cam_full.detach(),
                  torch.as_tensor(cam_centre).unsqueeze(0).to(device), keypoints)
    _, opt_j, opt_pose, opt_b, opt_cam_t, _ = res

    with torch.no_grad():
        smpl_out = smpl(betas=opt_b,
                        body_pose=opt_pose[:,3:],
                        global_orient=opt_pose[:,:3],
                        pose2rot=True,
                        transl=opt_cam_t)

    vertices = smpl_out.vertices[0].cpu().numpy()

    # ------------------------------------------------------------------
    # 4  Build the scene ------------------------------------------------
    # ------------------------------------------------------------------
    mesh = trimesh.Trimesh(vertices=vertices, faces=smpl.faces, process=False)
    global mesh_centroid, joints_centered
    mesh_centroid = mesh.centroid
    mesh.apply_translation(-mesh_centroid)

    joints_centered = smpl_out.joints[0].cpu().numpy() - mesh_centroid

    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh))

    # initial arrows ----------------------------------------------------
    _draw_arrows(scene)

    caption = [dict(
        text     = STATE_CONFIG[current_state]['text'],
        location = TextAlign.TOP_CENTER,
        font_name = r"C:\Windows\Fonts\arial.ttf",
        font_pt   = 30,
        color    = (0.,1.,0.,1.),
        scale    = 1.0)]

    key_handlers = {
        '1': (_update_state, [1]),
        '2': (_update_state, [2]),
        '3': (_update_state, [3]),
        '4': (_update_state, [4]),
    }

    pyrender.Viewer(scene,
                    use_raymond_lighting=True,
                    viewer_flags={'caption': caption},
                    registered_keys=key_handlers)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
