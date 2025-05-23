
import cv2
import numpy as np
from pyrender.trackball import Trackball
import trimesh
import pyrender
import visualizer
import json
from pyrender.constants import TextAlign


class demo:
    def __init__(self, vertices, faces, joints, smpl_segment):
        self._colormap = cv2.applyColorMap(np.arange(0, 256, 1, dtype=np.uint8), cv2.COLORMAP_AUTUMN)
        self._colormap = self._colormap[:, :, ::-1]
        self._colormap = np.dstack((self._colormap, 255 * np.ones(self._colormap.shape[0:2] + (1,), dtype=self._colormap.dtype)))

        self._vsv = visualizer.scene_manager(np.pi / 3.0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        self._vsv.set_smpl_mesh(mesh)
        self._solver = visualizer.solver(np.pi / 3.0, 1280, 720)
        self._solver.set_smpl_mesh(mesh, joints.T, smpl_segment)

        self._default_colors = self._vsv.get_mesh_colors().copy() #default color = [102 102 102 255] uint8
        self._focus_regions = [
            self._solver.focus_left_foot,
            self._solver.focus_right_foot,
            self._solver.focus_left_lower_leg,
            self._solver.focus_right_lower_leg,
            self._solver.focus_left_thigh,
            self._solver.focus_right_thigh,
            self._solver.focus_center_body,
            self._solver.focus_center_head,
            self._solver.focus_left_upper_arm,
            self._solver.focus_right_upper_arm,
        ]
        self._focus_regions_names = [
            'Left Foot',
            'Right Foot',
            'Left Lower Leg',
            'Right Lower Leg',
            'Left Thigh',
            'Right Thigh',
            'Body',
            'Head',
            'Left Upper Arm',
            'Right Upper Arm',
        ]
        self._region = len(self._focus_regions) - 1
        self._node_arrow = None
        self._radius = 0.05
        self._enable_face_filtering = True

    def update_surface_position(self, viewer):
        displacement, angle = self._position

        viewer.viewer_flags['caption'][1]['text'] = f'Position: [{displacement:.2f} m, {((angle/np.pi)*180):.2f} deg]'
        viewer.viewer_flags['caption'][2]['text'] = f'Pointer size: {self._radius:.3f} | Face filtering: {self._enable_face_filtering}'

        pose   = self._focus_solution[0]
        center = self._focus_solution[1]
        up     = pose[:3, 1:2]
        front  = pose[:3, 2:3]

        rotation = cv2.Rodrigues(up.reshape((-1,)) * angle)[0]

        center = center + up * displacement
        front  = rotation @ front

        point, face_index, vertex_index = self._solver.face_solver(center, front)

        if (point is not None):
            distances = self._solver.select_vertices(vertex_index, self._radius, np.Inf)
            selected_indices = distances.keys()
            if (self._enable_face_filtering):
                filtered_faces, filtered_indices = self._solver.select_complete_faces(selected_indices)
                if (len(filtered_indices) > 0):
                    selected_indices = filtered_indices
        else:
            distances = None
            selected_indices = None

        colors = self._default_colors.copy()
        if ((selected_indices is not None) and (distances is not None)):
            for vertex_index in selected_indices:
                colors[vertex_index, :] = self._colormap[min([int(255*(distances[vertex_index] / max([self._radius, 0.001]))),255]), 0, :]
        self._vsv.set_mesh_colors(colors)
        self._vsv.reload_mesh()

        if (self._node_arrow is not None):
            self._vsv._scene.remove_node(self._node_arrow)
            self._node_arrow = None

        if (point is not None):
            arrow = self.create_arrow((point + front*0.13).reshape((-1,)), (point + front*0.03).reshape((-1,)), 0.005, 0.015)
            if (arrow is not None):
                mat   = pyrender.MetallicRoughnessMaterial(baseColorFactor=(1, 0, 0, 1))
                self._node_arrow = self._vsv._scene.add(pyrender.Mesh.from_trimesh(arrow, material=mat, smooth=False))

    def focus_region(self, viewer):
        self._region = (self._region + 1) % len(self._focus_regions)
        self._focus_solution = self._focus_regions[self._region]()
        self._vsv.set_camera_pose(self._focus_solution[0])
        viewer._trackball = Trackball(self._focus_solution[0], viewer.viewport_size, 1.0, self._focus_solution[1].reshape((-1,)))
        viewer.viewer_flags['caption'][0]['text'] = f'Selected {self._focus_regions_names[self._region]}'
        viewer.viewer_flags['rotate_axis'] = self._focus_solution[0][:3, 1].reshape((-1,))
        viewer.viewer_flags['view_center'] = self._focus_solution[1].reshape((-1,))
        self._position = [0, 0]
        self.update_surface_position(viewer)
        
    def traverse(self, viewer, delta_displacement, delta_angle):
        displacement = self._position[0] + delta_displacement
        angle = self._position[1] + delta_angle
        if (angle > np.pi):
            angle = angle - (2*np.pi)
        if (angle < -np.pi):
            angle = angle + (2*np.pi)
        self._position = [displacement, angle]        
        self.update_surface_position(viewer)

    def adjust_pointer(self, viewer, delta_radius):
        radius = max([self._radius + delta_radius, 0])
        self._radius = radius
        self.update_surface_position(viewer)

    def toggle_filtering(self, viewer):
        self._enable_face_filtering = not self._enable_face_filtering
        self.update_surface_position(viewer)

    def run(self):
        caption = [
            dict(
                text     = 'Nothing selected',
                location = TextAlign.TOP_LEFT,
                font_name = r"C:\Windows\Fonts\arial.ttf",
                font_pt   = 30,
                color    = (0.,1.,0.,1.),
                scale    = 1.0),
            dict(
                text     = 'Position: [?,?]',
                location = TextAlign.CENTER_LEFT,
                font_name = r"C:\Windows\Fonts\arial.ttf",
                font_pt   = 30,
                color    = (0.,1.,0.,1.),
                scale    = 1.0),
            dict(
                text     = 'Pointer size: 0.05',
                location = TextAlign.BOTTOM_LEFT,
                font_name = r"C:\Windows\Fonts\arial.ttf",
                font_pt   = 30,
                color    = (0.,1.,0.,1.),
                scale    = 1.0),
            ]
        
        di = 0.02
        da = (10/180) * np.pi
        dr = 0.01

        key_handlers = {
            '1': (self.focus_region, []),
            '2': (self.traverse, [di, 0]),
            '3': (self.traverse, [-di, 0]),
            '4': (self.traverse, [0, da]),
            '5': (self.traverse, [0, -da]),
            '6': (self.adjust_pointer, [dr]),
            '7': (self.adjust_pointer, [-dr]),
            '8': (self.toggle_filtering, [])
        }

        pyrender.Viewer(self._vsv._scene,
                viewport_size=(1280,720),
                use_raymond_lighting=True,
                viewer_flags={'caption': caption},
                registered_keys=key_handlers)
    
    def create_arrow(self, start: np.ndarray, end: np.ndarray,
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


vertices = np.load('./data/test_vertices.npy')
faces = np.load('./data/test_faces.npy')
joints = np.load('./data/test_joints.npy')
with open('./data/smpl_vert_segmentation.json') as file:
    smpl_segment = json.load(file)

vis = demo(vertices, faces, joints, smpl_segment)
vis.run()

quit()

'''
smpl_segment
dict_keys([
'rightHand',
'rightUpLeg', [thigh]
'leftArm', 
'leftLeg', 
'leftToeBase', 
'leftFoot', 
'spine1', 
'spine2', 
'leftShoulder', 
'rightShoulder', 
'rightFoot', 
'head', 
'rightArm', 
'leftHandIndex1', 
'rightLeg', 
'rightHandIndex1', 
'leftForeArm', 
'rightForeArm', 
'neck', 
'rightToeBase', 
'spine', 
'leftUpLeg',  [thigh]
'leftHand', 
'hips'])
'''
