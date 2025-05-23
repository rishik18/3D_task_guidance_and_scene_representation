
import numpy as np
import pyrealsense2 as rs
import pyrender
import trimesh
import cv2
import collections

from pyglet import clock


class camera_opencv:
    def open(self, id=0, resolution=(640, 480), framerate=30):
        self._cap = cv2.VideoCapture(id)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS,          framerate)

    def read(self):
        return { 'color' : self._cap.read()[1] }

    def close(self):
        self._cap.release()


class camera_realsense:
    def open(self, resolution=(640, 480), framerate=30):
        self._pipe   = rs.pipeline()
        self._config = rs.config()

        self._config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16,  framerate)
        self._config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, framerate)
        
        self._profile = self._pipe.start(self._config)

        self._scale_depth = self._profile.get_device().first_depth_sensor().get_depth_scale()

        self._intrinsics_color = self._profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics_depth = self._profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        self._align = rs.align(rs.stream.color)
        self._pc = rs.pointcloud()

    def read(self, with_point_cloud=False):
        frames = self._align.process(self._pipe.wait_for_frames())

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        pointcloud  = np.asanyarray(self._pc.calculate(depth_frame).get_vertices()).view(np.float32).reshape(-1, 3) if (with_point_cloud) else None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return { 'color' : color_image, 'color_intrinsics' : self._intrinsics_color, 'depth' : depth_image, 'depth_intrinsics' : self._intrinsics_depth, 'depth_scale' : self._scale_depth, 'pointcloud' : pointcloud }

    def close(self):
        self._pipe.stop()


class viewer(pyrender.Viewer):
    @staticmethod
    def _time_event(dt, self):
        if self._is_active:
            self.on_begin_time_event()
        pyrender.Viewer._time_event(dt, self)
        if self._is_active:
            self.on_end_time_event()

    def switch_to(self):
        if (not self.__time_event_ff):
            clock.unschedule(pyrender.Viewer._time_event)
            clock.schedule_interval(viewer._time_event, 1.0 / self.viewer_flags['refresh_rate'], self)
            self.__time_event_ff = True
        super().switch_to()

    def __init__(self, *args, **kwargs):
        self.__time_event_ff = False
        super().__init__(*args, **kwargs)

    def set_camera_target(self, pose, axis, center):
        self._trackball._n_pose = pose
        self._trackball._n_target = center.reshape((-1,))
        self.viewer_flags['rotate_axis'] = axis.reshape((-1,))
        self.viewer_flags['view_center'] = center.reshape((-1,))

    def _invoke(self, tup):
        if (tup):
            callback = None
            args = []
            kwargs = {}
            if not isinstance(tup, (list, tuple, np.ndarray)):
                callback = tup
            else:
                callback = tup[0]
                if len(tup) == 2:
                    args = tup[1]
                if len(tup) == 3:
                    kwargs = tup[2]
            callback(self, *args, **kwargs)
        
    def on_begin_time_event(self):
        self._invoke(self.viewer_flags.get('vsv.hook_on_begin'))
        
    def on_end_time_event(self):
        self._invoke(self.viewer_flags.get('vsv.hook_on_end'))


class scene_manager:
    def __init__(self, camera_yfov):
        self._scene  = pyrender.Scene()
        self._camera = pyrender.PerspectiveCamera(yfov=camera_yfov)
        self._light  = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        
        self._camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float32).reshape((4, 4))
        
        self._node_camera = self._scene.add(self._camera, pose=self._camera_pose)
        self._node_light  = self._scene.add(self._light,  pose=self._camera_pose)
        self._node_mesh   = None
        self._node_groups = {}

    def set_smpl_mesh(self, mesh):
        self._mesh = mesh
        self.reload_mesh()

    def get_mesh_colors(self):
        return self._mesh.visual.vertex_colors
    
    def set_mesh_colors(self, colors):
        self._mesh.visual.vertex_colors = colors

    def reload_mesh(self):
        if (self._node_mesh is not None):
            self._scene.remove_node(self._node_mesh)
            self._node_mesh = None
        self._node_mesh = self._scene.add(pyrender.Mesh.from_trimesh(self._mesh))

    def get_camera_pose(self):
        return self._camera_pose
    
    def set_camera_pose(self, camera_pose):
        self._camera_pose = camera_pose

        self._scene.set_pose(self._node_camera, self._camera_pose)
        self._scene.set_pose(self._node_light,  self._camera_pose)

    def clear_group(self, group):
        nodes = self._node_groups.get(group, [])
        for node in nodes:
            self._scene.remove_node(node)
        self._node_groups[group] = []

    def add(self, group, object):
        nodes = self._node_groups.get(group, [])
        nodes.append(self._scene.add(object))
        self._node_groups[group] = nodes


class solver:
    def __init__(self, camera_yfov, viewport_width, viewport_height):
        self._camera_fy   = viewport_height / (2 * np.tan(0.5 * camera_yfov))
        self._camera_fx   = self._camera_fy
        self._camera_cy   = viewport_height / 2
        self._camera_cx   = viewport_width / 2
        self._camera_yfov = camera_yfov
        self._camera_xfov = 2 * np.arctan(viewport_width / (2 * self._camera_fx))

    def set_smpl_mesh(self, mesh, joints, segmentation):
        self._mesh = mesh
        self._joints = joints
        self._segmentation = segmentation

    def get_joint(self, index):
        return self._joints[:, index].reshape((3, -1))
    
    def cross(self, a, b):
        return np.cross(a.reshape((-1)), b.reshape((-1))).reshape((3, -1))
    
    def normalize(self, a):
        return a / np.linalg.norm(a)

    def focus_solver(self, x, y, z, center, points):
        p = np.eye(4, 4, dtype=np.float32)

        p[:3, 0:1] = x
        p[:3, 1:2] = y
        p[:3, 2:3] = z

        dp = (points - center)
        dx = np.abs(x.T @ dp)
        dy = np.abs(y.T @ dp)
        dz = -z.T @ dp
        wx = (dx / np.tan(self._camera_xfov / 2)) - dz
        wy = (dy / np.tan(self._camera_yfov / 2)) - dz
        wz = np.max(np.hstack((wy, wx)))

        p[:3, 3:4] = (center + wz * z)

        return p, center, wz
    
    def face_solver(self, origin, direction):
        point, rid, tid = self._mesh.ray.intersects_location(origin.T, direction.T, multiple_hits=False)
        if (len(rid) <= 0):
            return (None, None, None)
        point = point.T
        face_index = tid[0]
        vertex_indices = self._mesh.faces.view(np.ndarray)[face_index].tolist()
        vertices = self._mesh.vertices.view(np.ndarray)[vertex_indices, :].T
        distances = np.linalg.norm(point - vertices, axis=0)
        snap_index = np.argmin(distances)

        return point, face_index, vertex_indices[snap_index]
    
    def select_vertices(self, origin_vertex_index, radius, level):
        vertices  = self._mesh.vertices.view(np.ndarray)
        neighbors = self._mesh.vertex_neighbors
        buffer    = collections.deque()
        distances = {origin_vertex_index : 0}

        buffer.append((origin_vertex_index, 0, 0))

        while (len(buffer) > 0):
            vertex_index, vertex_distance, vertex_level = buffer.popleft()
            vertex_xyz = vertices[vertex_index, :]

            for neighbor_index in neighbors[vertex_index]:
                neighbor_xyz = vertices[neighbor_index, :]
                neighbor_distance = vertex_distance + np.linalg.norm(neighbor_xyz - vertex_xyz)
                neighbor_level = vertex_level + 1
                if ((neighbor_distance <= radius) and (neighbor_level <= level) and (neighbor_distance < distances.get(neighbor_index, np.Inf))):          
                    buffer.append((neighbor_index, neighbor_distance, neighbor_level))
                    distances[neighbor_index] = neighbor_distance
        
        return distances
    
    def select_complete_faces(self, vertex_indices):
        vertex_faces = self._mesh.vertex_faces
        faces = self._mesh.faces.view(np.ndarray)
        face_indices_complete = set()
        vertex_indices_complete = set()
        vertex_indices_seen = set()

        for vertex_index in vertex_indices:
            if (vertex_index in vertex_indices_seen):
                continue
            for face_index in vertex_faces[vertex_index, :]:
                if (face_index >= 0):
                    face_vertices = faces[face_index].tolist()
                    vertex_indices_seen.update(face_vertices)
                    keep = all([face_vertex in vertex_indices for face_vertex in face_vertices])
                    if (keep):
                        face_indices_complete.add(face_index)
                        vertex_indices_complete.update(face_vertices)
        
        return list(face_indices_complete), list(vertex_indices_complete)
    
    # Single -------------------------------------------------------------------

    def focus_foot(self, bigtoe, smalltoe, ankle, heel):
        left  = self.cross(ankle - heel, bigtoe - ankle)
        front = self.cross(left, ankle - smalltoe)
        up    = self.cross(front, left)

        left  = self.normalize(left)
        front = self.normalize(front)
        up    = self.normalize(up)

        center = (ankle + bigtoe) * 0.5
        points = np.hstack((bigtoe, smalltoe, ankle, heel))

        return self.focus_solver(left, up, front, center, points)

    def focus_lower_leg(self, bigtoe, ankle, knee):
        up    = knee - ankle
        left  = self.cross(up, bigtoe - ankle)
        front = self.cross(left, up)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (ankle + knee) * 0.5
        points = np.hstack((bigtoe, ankle, knee))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_thigh(self, ankle, knee, hip):
        up    = hip - knee
        left  = self.cross(up, knee - ankle)
        front = self.cross(left, up)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (hip + knee) * 0.5
        points = np.hstack((knee, hip))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_body(self, lhip, mhip, rhip, neck):
        left  = lhip - rhip
        front = self.cross(left, neck - mhip)
        up    = self.cross(front, left)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = (mhip + neck) * 0.5
        points = np.hstack((lhip, mhip, rhip, neck))
                
        return self.focus_solver(left, up, front, center, points)
    
    def focus_head(self, lear, rear, neck, nose):
        left  = lear - rear
        up    = lear - neck
        front = self.cross(left, up)
        up    = self.cross(front, left)

        left  = self.normalize(left)
        up    = self.normalize(up)
        front = self.normalize(front)

        center = (nose + lear + rear) / 3
        points = np.hstack((lear, rear, neck, nose))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_upper_arm(self, wrist, elbow, shoulder):
        up    = shoulder - elbow
        left  = self.cross(up, wrist - elbow)
        front = self.cross(left, up)

        left  = self.normalize(left)
        up    = self.normalize(up)
        front = self.normalize(front)

        center = (elbow + shoulder) * 0.5
        points = np.hstack((shoulder, elbow))        

        return self.focus_solver(left, up, front, center, points)

    def focus_left_foot(self):
        bigtoe   = self.get_joint(19)
        smalltoe = self.get_joint(20)
        ankle    = self.get_joint(14)
        heel     = self.get_joint(21)

        return self.focus_foot(bigtoe, smalltoe, ankle, heel)
    
    def focus_right_foot(self):
        bigtoe   = self.get_joint(22)
        smalltoe = self.get_joint(23)
        ankle    = self.get_joint(11)
        heel     = self.get_joint(24)

        return self.focus_foot(bigtoe, smalltoe, ankle, heel)
    
    def focus_left_lower_leg(self):
        bigtoe = self.get_joint(19)
        ankle  = self.get_joint(14)
        knee   = self.get_joint(13)

        return self.focus_lower_leg(bigtoe, ankle, knee)

    def focus_right_lower_leg(self):
        bigtoe = self.get_joint(22)
        ankle  = self.get_joint(11)
        knee   = self.get_joint(10)

        return self.focus_lower_leg(bigtoe, ankle, knee)
    
    def focus_left_thigh(self):
        ankle = self.get_joint(14)
        knee  = self.get_joint(13)
        hip   = self.get_joint(12)

        return self.focus_thigh(ankle, knee, hip)
    
    def focus_right_thigh(self):
        ankle = self.get_joint(11)
        knee  = self.get_joint(10)
        hip   = self.get_joint(9)

        return self.focus_thigh(ankle, knee, hip)
    
    def focus_center_body(self):
        lhip = self.get_joint(12)
        mhip = self.get_joint(8)
        rhip = self.get_joint(9)
        neck = self.get_joint(1)

        return self.focus_body(lhip, mhip, rhip, neck)
    
    def focus_center_head(self):
        lear = self.get_joint(18)
        rear = self.get_joint(17)
        neck = self.get_joint(1)
        nose = self.get_joint(0)

        return self.focus_head(lear, rear, neck, nose)
    
    def focus_left_upper_arm(self):
        wrist    = self.get_joint(7)
        elbow    = self.get_joint(6)
        shoulder = self.get_joint(5)

        return self.focus_upper_arm(wrist, elbow, shoulder)
    
    def focus_right_upper_arm(self):
        wrist    = self.get_joint(4)
        elbow    = self.get_joint(3)
        shoulder = self.get_joint(2)

        return self.focus_upper_arm(wrist, elbow, shoulder)
    
    # Composite ----------------------------------------------------------------

    def focus_whole(self, lhip, mhip, rhip, neck, lear, rear, rsmalltoe, lsmalltoe):
        left  = lhip - rhip
        front = self.cross(left, neck - mhip)
        up    = self.cross(front, left)

        up    = self.normalize(up)
        left  = self.normalize(left)
        front = self.normalize(front)

        center = mhip
        points = np.hstack((lhip, mhip, rhip, neck, lear, rear, rsmalltoe, lsmalltoe))
                
        return self.focus_solver(left, up, front, center, points)
        
    def focus_leg(self, ankle, knee, hip, bigtoe):
        up_thigh        = hip - knee
        up_lower_leg    = knee - ankle
        left            = self.cross(up_thigh, up_lower_leg)
        front_thigh     = self.cross(left, up_thigh)
        front_lower_leg = self.cross(left, up_lower_leg)

        left  = self.normalize(left)
        front = self.normalize(self.normalize(front_thigh) + self.normalize(front_lower_leg))
        up    = self.normalize(self.cross(front, left))
        front = self.normalize(self.cross(left, up))

        center = knee
        points = np.hstack((ankle, knee, hip, bigtoe))

        return self.focus_solver(left, up, front, center, points)
    
    def focus_right_leg(self):
        ankle  = self.get_joint(11)
        knee   = self.get_joint(10)
        hip    = self.get_joint(9)
        bigtoe = self.get_joint(22)

        return self.focus_leg(ankle, knee, hip, bigtoe)

    def focus_left_leg(self):
        ankle  = self.get_joint(14)
        knee   = self.get_joint(13)
        hip    = self.get_joint(12)
        bigtoe = self.get_joint(19)

        return self.focus_leg(ankle, knee, hip, bigtoe)
    
    def focus_center_whole(self):
        lhip      = self.get_joint(12)
        mhip      = self.get_joint(8)
        rhip      = self.get_joint(9)
        neck      = self.get_joint(1)
        lear      = self.get_joint(18)
        rear      = self.get_joint(17)
        rsmalltoe = self.get_joint(23)
        lsmalltoe = self.get_joint(20)

        return self.focus_whole(lhip, mhip, rhip, neck, lear, rear, rsmalltoe, lsmalltoe)



        
        
        
        









    
    



    '''
    def render(self):
        self._color, self._depth = self._renderer.render(self._scene)
        self._color = self._color.copy()

        return (self._color, self._depth)
    '''
    #self._renderer = pyrender.OffscreenRenderer(viewport_width, viewport_height)
    '''
    def project_points(self, points):
        r = self._camera_pose[:3, :3]
        t = self._camera_pose[:3, 3:4]

        camera_points     = r.T @ points - r.T @ t
        normalized_points = camera_points[0:2, :] / -camera_points[2, :]

        image_x =  normalized_points[0, :] * self._camera_fx + self._camera_cx
        image_y = -normalized_points[1, :] * self._camera_fy + self._camera_cy

        return np.vstack((image_x, image_y))
    '''
    # 

    
    
    
    


    











        
  


    '''
          REar-REye--------|--------LEye-LEar
                         Nose
          RShoulder------Neck-------LShoulder
             RElbow        |        LElbow
             RWrist        |        LWrist
                           |
               RHip-----MidHip------LHip
              RKnee                 LKnee
             RAnkle                 LAnkle
    RSmallToe RHeel RBigToe LBigToe LHeel LSmallToe
    '''
    '''
    'OP Nose', # 0
    'OP Neck', # 1
    'OP RShoulder', # 2
    'OP RElbow', # 3
    'OP RWrist', # 4
    'OP LShoulder', # 5
    'OP LElbow', # 6
    'OP LWrist', # 7
    'OP MidHip', # 8
    'OP RHip', # 9
    'OP RKnee', # 10
    'OP RAnkle', # 11
    'OP LHip', # 12
    'OP LKnee', # 13
    'OP LAnkle', # 14
    'OP REye', # 15
    'OP LEye', # 16
    'OP REar', # 17
    'OP LEar', # 18
    'OP LBigToe', # 19
    'OP LSmallToe', # 20
    'OP LHeel', # 21
    'OP RBigToe', # 22
    'OP RSmallToe', # 23
    'OP RHeel', # 24
    '''  

# NO: small toe, heel
# foot side in: (heel-ankle) X (bigtoe-ankle) !
# lower leg side in: (bigtoe-ankle) x (knee-ankle)
# thigh side in: (ankle-knee) X (knee-hip)
# body front: (Rhip-Lhip) X (Lhip-neck)
# face front: (LEar-REar) X (REar-neck)
# upper arm side in: (wrist-elbow) X (shoulder-elbow)
# lower arm: no
# hand: no