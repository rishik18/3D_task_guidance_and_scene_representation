# marker_gui.py
"""Interactive SMPL-X viewer with GUI controls to place a custom anatomical marker.

Requirements
------------
    pip install open3d==0.18 smplx torch trimesh numpy

Run
---
    python marker_gui.py --model-folder /path/to/models --model-type smplx
"""

import argparse
import math
import os.path as osp
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import trimesh
import smplx
from open3d.visualization.rendering import Open3DScene

# ────────────────────────────────────────────────────────────────────────────────
# Geometry helper: place a marker on the mesh given two joints, a distance *l*   
# from joint A toward B, and an azimuth α in the joint‑local X‑Z plane.           
# Returns the hit point in world coordinates or *None* if the ray misses.        
# ────────────────────────────────────────────────────────────────────────────────

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

# Undirected edges of the 24-joint SMPL skeleton
SMPL_EDGES = [
    (0, 1), (1, 4), (4, 7), (7, 10),           # left leg
    (0, 2), (2, 5), (5, 8), (8, 11),           # right leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # spine & head
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),   # left arm
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23)    # right arm
]

# Build an adjacency list:  idx → [neighbour-indices]
ADJACENT = {i: [] for i in range(24)}
for a, b in SMPL_EDGES:
    ADJACENT[a].append(b)
    ADJACENT[b].append(a)




def add_custom_marker_az(mesh_trimesh: trimesh.Trimesh,
                         joints: np.ndarray,
                         joint_id_a: int,
                         joint_id_b: int,
                         l: float,
                         alpha_deg: float) -> Optional[np.ndarray]:
    """Ray‑cast from *A* toward *B* and return intersection on the mesh.

    Parameters
    ----------
    mesh_trimesh : trimesh.Trimesh
    joints       : (N, 3) ndarray of joint locations (world)
    joint_id_a   : int  • origin joint (point A)
    joint_id_b   : int  • articulation joint (point B)
    l            : float
        Offset from A toward B in metres.
    alpha_deg    : float
        Azimuth inside the local X‑Z plane: 0°→+X, 90°→+Z.
    """
    pA, pB = joints[joint_id_a], joints[joint_id_b]
    vAB = pB - pA
    normAB = np.linalg.norm(vAB)
    if normAB < 1e-8:
        return None

    # Orthonormal frame centred on *A* with +Y along AB
    y_axis = vAB / normAB
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, y_axis)) > 0.95:  # y_axis almost Z → choose different up
        world_up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(world_up, y_axis)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(y_axis, x_axis)

    #origin = pA + (l / normAB) * vAB
    origin = pA + l * vAB
    alpha = math.radians(alpha_deg)
    dir_vec = math.cos(alpha) * x_axis + math.sin(alpha) * z_axis

    loc, *_ = mesh_trimesh.ray.intersects_location(origin.reshape(1, 3),
                                                   dir_vec.reshape(1, 3),
                                                   multiple_hits=False)
    return loc[0] if len(loc) else None


# ────────────────────────────────────────────────────────────────────────────────
# Main GUI application                                                            
# ────────────────────────────────────────────────────────────────────────────────

class MarkerApp:
    def __init__(self, model_folder: str, model_type: str = "smpl", gender: str = "neutral",
                 num_betas: int = 10, num_expression_coeffs: int = 10):
        self._marker_geom_name = "marker"
        self._marker_material = rendering.MaterialRecord()
        self._marker_material.shader = "defaultUnlit"
        self._marker_material.base_color = (0.2, 0.8, 1.0, 1.0)

        # ---------------- Load SMPL‑X body ------------------------------------
        model = smplx.create(model_folder, model_type=model_type, gender=gender,
                             num_betas=num_betas, num_expression_coeffs=num_expression_coeffs,
                             ext="npz")
        betas = torch.randn(1, model.num_betas)
        expression = torch.randn(1, model.num_expression_coeffs)
        output = model(betas=betas, expression=expression, return_verts=True)
        self.joints = output.joints.detach().cpu().numpy().squeeze()

        self.joint_names = SMPL_JOINT_NAMES[:24]
        verts = output.vertices.detach().cpu().numpy().squeeze()
        self.tri_mesh = trimesh.Trimesh(verts, model.faces, process=False)

        # Convert to Open3D
        self.o3d_body = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(model.faces))
        self.o3d_body.compute_vertex_normals()
        self.o3d_body.paint_uniform_color([0.7, 0.7, 0.7])

        body_mat = rendering.MaterialRecord()
        body_mat.shader      = "defaultLit"              # opaque PBR shader
        body_mat.base_color  = (0.7, 0.7, 0.7, 1.0)      # RGBA, alpha = 1
        body_mat.base_roughness   = 0.9                       # nice matte look
        body_mat.base_metallic    = 0.0
        body_mat.base_reflectance = 0.04                 # dielectric

        joint_sphere_mat = rendering.MaterialRecord()
        joint_sphere_mat.shader      = "defaultLit"              # opaque PBR shader
        joint_sphere_mat.base_color  = (0.9, 0.0, 0.0, 1.0)      # RGBA, alpha = 1
        joint_sphere_mat.base_roughness   = 0.9                       # nice matte look
        joint_sphere_mat.base_metallic    = 0.0
        joint_sphere_mat.base_reflectance = 0.04                 # dielectric

        # Build small spheres for joints (optional visual aid)
        self.joint_spheres = []
        sph_template = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sph_template.paint_uniform_color([0.9, 0.1, 0.1])
        for j in self.joints:
            s = sph_template.translate(j, relative=False)
            self.joint_spheres.append(s)
            sph_template = sph_template.translate(-j, relative=False)  # reset

        # ---------------- Build GUI ------------------------------------------
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window("Custom Marker GUI", 1280, 720)

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.add_geometry("body", self.o3d_body, body_mat)
        #for idx, js in enumerate(self.joint_spheres):
        #    self.scene_widget.scene.add_geometry(f"j{idx}", js, joint_sphere_mat)

        # Set up camera -------------------------------------------------------
        bounds = self.o3d_body.get_axis_aligned_bounding_box()
        centre = bounds.get_center()
        self.scene_widget.scene.camera.look_at(centre, centre + np.array([0, 1, 0.2]), np.array([0, 0, 1]))
        # Explicit perspective projection: add FovType argument (Open3D ≥0.17)
        self.scene_widget.scene.camera.set_projection(60.0,          # vertical FOV (deg)
                                                       1.0,           # aspect (will auto‑update on resize)
                                                       0.01,          # near
                                                       10.0,          # far
                                                       rendering.Camera.FovType.Vertical)
        sun_dir = (0.577, -0.577, -0.577)          # unit-length direction vector
        self.scene_widget.scene.set_lighting(
                Open3DScene.LightingProfile.MED_SHADOWS,
                sun_dir)
        # ---------- Control panel --------------------------------------------
        self.panel = gui.Vert(0, gui.Margins(10, 10, 10, 10))
        self.panel.add_child(gui.Label("Joint A (origin)"))
        self.combo_a = gui.Combobox()
        self.panel.add_child(self.combo_a)
        self.panel.add_child(gui.Label("Joint B (direction)"))
        self.combo_b = gui.Combobox()
        self.panel.add_child(self.combo_b)

        self.length_slider = gui.Slider(gui.Slider.DOUBLE)
        self.length_slider.set_limits(0.0, 1.0)
        self.length_slider.double_value = 0.1
        self.panel.add_child(gui.Label("% of length along A to B"))
        self.panel.add_child(self.length_slider)

        self.angle_slider = gui.Slider(gui.Slider.DOUBLE)
        self.angle_slider.set_limits(0.0, 360.0)
        self.angle_slider.double_value = 90.0
        self.panel.add_child(gui.Label("Azimuth (deg)"))
        self.panel.add_child(self.angle_slider)

        self.button = gui.Button("Place marker")
        self.button.set_on_clicked(self._on_update_marker)
        self.panel.add_child(self.button)


        # Populate joint lists
        for name in self.joint_names:
            self.combo_a.add_item(name)
        self.combo_a.selected_index = 5      # right_knee
        self._on_joint_a_changed(self.joint_names[5], 5)           # populate Joint B once at start
        self.combo_a.set_on_selection_changed(self._on_joint_a_changed)

        # Layout
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

        gui.Application.instance.run()
    def _on_joint_a_changed(self, _text, new_index):
        # • Clear the options in combo_b

        self.combo_b.clear_items()

        # • Re-add only neighbours of the newly selected joint
        for j_idx in ADJACENT.get(new_index, []):
            self.combo_b.add_item(self.joint_names[j_idx])
        # • Default B to the first available neighbour (if any)
        if self.combo_b.number_of_items > 0:
            self.combo_b.selected_index = 0

    # ────────── Callbacks ────────────────────────────────────────────────────
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        side = 300  # px width of control panel
        self.panel.frame = gui.Rect(r.x, r.y, side, r.height)
        self.scene_widget.frame = gui.Rect(r.x + side, r.y, r.width - side, r.height)

    def _on_update_marker(self):
        a_id = self.combo_a.selected_index
        b_id = SMPL_JOINT_NAMES.index(self.combo_b.selected_text)
        l = self.length_slider.double_value
        alpha = self.angle_slider.double_value

        hit = add_custom_marker_az(self.tri_mesh, self.joints, a_id, b_id, l, alpha)
        if hit is None:
            gui.msgbox("Ray did not hit the mesh. Try different parameters.")
            return

        # Remove previous marker if exists
        if self.scene_widget.scene.has_geometry(self._marker_geom_name):
            self.scene_widget.scene.remove_geometry(self._marker_geom_name)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(hit, relative=False)
        sphere.paint_uniform_color([0.2, 0.8, 1.0])
        self.scene_widget.scene.add_geometry(self._marker_geom_name, sphere, self._marker_material)


# ────────────────────────────────────────────────────────────────────────────────
# Entry point                                                                     
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive SMPL‑X marker GUI")
    parser.add_argument("--model-folder", default=".\data", help="Path to SMPL‑X model directory")
    parser.add_argument("--model-type", default="smpl", choices=["smpl", "smplh", "smplx"], help="Type of model")
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    args = parser.parse_args()

    MarkerApp(args.model_folder, model_type=args.model_type, gender=args.gender)
