"""
Copyright (C) 2021 MICC-Unifi.  All rights reserved.
The script performs data preprocessing:
- Rotates the pointclouds to make them frontal
- Detects the nosetip
- Crops the face region
It can be applied to both 3D meshes and depth images from a Kinect sensor
"""

from denseRegistration import *
import pywavefront

HD = True
params = {'debug': True, 'writePc': False}

if not HD:
    # Depth camera intrinsics from MICC Kinect. Can be changed to any Kinect params
    kp = [366.4027, 366.4027, 261.1273, 208.4769, -0.2712, 0.0915, 0.0922]
    # Face region to crop
    radius = 0.12
    # Z-threshold for the face region
    face_thresh = -0.15
    # Angle to rotate the pointcloud (x-axis)
    rot_angle = 5
    # Path to the depth frames
    frame_path = 'Ex-frames/depth.png'
    processed_path = 'Ex-frames/depth.png'
    savepath = 'Ex-frames/depth_proc.ply'

    # Convert depth to meters
    D_in_metres, depth_original = depth_to_metres(frame_path)
    # Get the pointcloud from converted depth
    P3D = depth_in_metres_calibrator(D_in_metres, kp, params['debug'])
    # Create open3D obj
    P3D_pc = o3d.geometry.PointCloud()
    P3D_pc.points = o3d.utility.Vector3dVector(P3D)

    if params['debug']:
        draw_point_cloud(P3D_pc)

    # isolating face from torso.................................................
    pcd_array = np.asarray(P3D_pc.points)
    pcd_foreground_face = pcd_array[pcd_array[:, 1] > face_thresh, :]

    # rotate to compensate small pitch rotation
    pcd_foreground_face = rotate_ndarray_3d(pcd_foreground_face, rot_angle, [1, 0, 0])

    if params['debug']:
        P3D_pc_fore_f = o3d.geometry.PointCloud()
        P3D_pc_fore_f.points = o3d.utility.Vector3dVector(pcd_foreground_face)
        draw_point_cloud(P3D_pc_fore_f)

    # find nose tip and save pointCloud of landmark and face......................
    lm_nose_tip = find_nose_tip_with_landmark(pcd_foreground_face)

    if params['debug']:
        draw_point_cloud_with_landmarks(P3D_pc_fore_f, lm_nose_tip, 'Nose in Red P3D')
    # ............................................................

    # Crop the face ...................................
    lm_nose_tip_points = np.asarray(lm_nose_tip.points)
    diffvertex = pcd_foreground_face - lm_nose_tip_points[0, :]
    dist = np.power(np.sum(np.power(diffvertex.T, 2), axis=0), 0.5)

    idxInsideBall = dist < radius
    idxInsideBall = idxInsideBall * 1
    idxInsideBall = np.where(idxInsideBall)
    pcd_foreground_face = pcd_foreground_face[idxInsideBall, :]
    pcd_foreground_face = np.squeeze(pcd_foreground_face)

    lm_nose_tip = find_nose_tip_with_landmark(pcd_foreground_face)
    # .....................................................

    pcd_f_face = o3d.geometry.PointCloud()
    pcd_f_face.points = o3d.utility.Vector3dVector(pcd_foreground_face)
    pcd_f_face_array = np.asarray(pcd_f_face.points)

    o3d.io.write_point_cloud(savepath, pcd_f_face)

    if params['debug']:
        draw_point_cloud_with_landmarks(pcd_f_face, lm_nose_tip, 'Lm post crop')
else:
    # SCAN HD ------------------------------------------------------------
    # take the .obj point cloud in meshlab and save it as .ply and load it

    pc_path = 'Ex-pc/1.obj'
    savepathP = 'Ex-pc/1_processed.ply'
    savepathR = 'Ex-pc/1_rotated.ply'
    # Angle to rotate the pointcloud (x-axis)
    rot_angle = -25
    # Radius to crop with center at the nosetip
    radius = 105

    objfile = pywavefront.Wavefront(pc_path, create_materials=True)
    start_pc_rgb_array = np.asarray(objfile.vertices)

    start_pc_rgb = o3d.geometry.PointCloud()
    start_pc_rgb.points = o3d.utility.Vector3dVector(start_pc_rgb_array)

    if params['debug']:
        draw_point_cloud(start_pc_rgb)

    # Rotate to make it front-facing
    start_pc_rgb_array = rotate_ndarray_3d(start_pc_rgb_array, rot_angle, [1, 0, 0])

    start_pc_rgb = o3d.geometry.PointCloud()
    start_pc_rgb.points = o3d.utility.Vector3dVector(start_pc_rgb_array)
    # Save rotated mesh
    o3d.io.write_point_cloud(savepathR, start_pc_rgb)

    if params['debug']:
        draw_point_cloud(start_pc_rgb)

    pcd_foreground_face_rgb = start_pc_rgb_array

    lm_nose_tip_rgb = find_nose_tip_with_landmark(pcd_foreground_face_rgb)

    # Crop the face ...................................
    lm_nose_tip_points = lm_nose_tip_rgb.points
    lm_nose_tip_points = np.asarray(lm_nose_tip_points)
    diffvertex = pcd_foreground_face_rgb - lm_nose_tip_points[0, :]
    dist = np.power(np.sum(np.power(diffvertex.T, 2), axis=0), 0.5)
    idxInsideBall = dist < radius
    idxInsideBall = idxInsideBall * 1
    idxInsideBall = np.where(idxInsideBall)
    pcd_foreground_face_rgb = pcd_foreground_face_rgb[idxInsideBall, :]
    pcd_foreground_face_rgb = np.squeeze(pcd_foreground_face_rgb)
    # .....................................................

    start_pc_rgb = o3d.geometry.PointCloud()
    start_pc_rgb.points = o3d.utility.Vector3dVector(pcd_foreground_face_rgb)
    o3d.io.write_point_cloud(savepathP, start_pc_rgb)

    if params['debug']:
        draw_point_cloud(start_pc_rgb)

