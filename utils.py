import open3d as o3d
import copy
import numpy as np
from scipy.spatial.transform import Rotation as R

'''
Functions for visualization
'''


def draw_point_cloud_with_landmarks(pointCloud, landmarks, window_name):
    pointCloud.paint_uniform_color([0, 0, 1])
    landmarks.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pointCloud, landmarks], window_name)


def draw_point_cloud(source):
    source.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


'''
Functions for meshes
'''


def rotate_ndarray_3d(input_array, degree, axes_array):
    rotation_degrees = degree
    rotation_radians = np.radians(rotation_degrees)
    rotation_axis = np.array(axes_array)

    rotation_vector = rotation_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(input_array)
    return rotated_vec


def find_nose_tip_with_landmark(source):
    nose_tip_index = np.argmax(source[:, 2])
    nose_tip = source[nose_tip_index, :]

    nose_tip = np.concatenate((nose_tip, nose_tip))
    nose_tip = np.concatenate((nose_tip, nose_tip)).reshape(4, 3)

    lm_nose_tip = o3d.geometry.PointCloud()
    lm_nose_tip.points = o3d.utility.Vector3dVector(nose_tip)

    return lm_nose_tip


'''
Functions for depth images to pc
'''

def create_pc_from_depth_img(img_path):
    depth_raw = o3d.io.read_image(img_path)
    pcd = o3d.create_point_cloud_from_depth_image(depth_raw, o3d.PinholeCameraIntrinsic(
        o3d.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault), np.eye(4), 100, 100, 1)
    return pcd


def depth_to_metres(frame):
    image = o3d.io.read_image(frame)
    image = np.asarray(image)
    depth_m = 1.0 / (image * -0.0030711016 + 3.3309495161)

    return depth_m, image


def depth_in_metres_calibrator(D, kp, deb):
    K = [kp[0], 0, kp[2], 0, kp[1], kp[3], 0, 0, 1]
    K = np.reshape(K, (3, 3))
    X, Y = np.meshgrid(np.arange(1, np.size(D, axis=1) + 1), np.arange(1, np.size(D, axis=0) + 1))
    Y3d = ((Y[:] - K[1, 2]) * D[:]) / K[1, 1]
    X3d = ((X[:] - K[0, 2]) * D[:]) / K[0, 0]
    Z3d = D[:]
    X3d = X3d.flatten('F')
    Y3d = Y3d.flatten('F')
    Z3d = Z3d.flatten('F')
    scaled_vec = np.dstack([X3d, Y3d, Z3d])
    scaled_vec = np.squeeze(scaled_vec)

    scaled_vec_pc = o3d.geometry.PointCloud()
    scaled_vec_pc.points = o3d.utility.Vector3dVector(scaled_vec)
    # o3d.io.write_point_cloud("3D point clouds/DiMC_C_TEST_scaled_vec.ply", scaled_vec_pc)
    if deb:
        draw_point_cloud(scaled_vec_pc)

    scaled_vec = scaled_vec[scaled_vec[:, 2] > 0.5, :]

    scaled_vec_pc = o3d.geometry.PointCloud()
    scaled_vec_pc.points = o3d.utility.Vector3dVector(scaled_vec)
    # o3d.io.write_point_cloud("3D point clouds/DiMC_C_TEST_scaled_vec.ply", scaled_vec_pc)

    if deb:
        draw_point_cloud(scaled_vec_pc)

    scaled_vec = rotate_ndarray_3d(scaled_vec, 185, [1, 0, 0])

    scaled_vec_pc = o3d.geometry.PointCloud()
    scaled_vec_pc.points = o3d.utility.Vector3dVector(scaled_vec)
    # o3d.io.write_point_cloud("3D point clouds/DiMC_C_TEST_scaled_vec.ply", scaled_vec_pc)
    if deb:
        draw_point_cloud(scaled_vec_pc)

    return scaled_vec