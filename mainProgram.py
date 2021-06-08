from contextlib import redirect_stdout
import pyvista as pv
from denseRegistration import *
import pywavefront
import glob
import os

HD = False

if not HD:
    params = {'debug': True, 'writePc': False}

    frame_number = "4631"
    print('Computing depth frame ' + frame_number)
    # depth camera intrinsics from MICC Kinect
    kp = [366.4027, 366.4027, 261.1273, 208.4769, -0.2712, 0.0915, 0.0922]
    radius = 0.12
    face_thresh = -0.15
    rot_angle = 15

    LR_path = '/mnt/storage2-data-2T/LR-HR-MICC-dataset/LR/'
    subj_list = glob.glob(LR_path + '*')

    for s in subj_list:
        seqlist = glob.glob(s + '/depth_frames/*')  # exclude txt
        print('Doing subject {}'.format(s))
        for m in seqlist:
            print('Doing Sequence {}'.format(m))
            frames_list = sorted(glob.glob(m + '/*'))
            frame_number = frames_list[-5]  # Choose one frame

            processed_path = '/'.join(frame_number.split('/')[:-3]) + '/processed/'

            if not os.path.exists(processed_path):
                os.mkdir(processed_path)

            savepath = processed_path + frame_number.split('/')[-2] +'_cropped.ply'

            if os.path.isfile(savepath):
                continue

            D_in_metres, depth_original = depth_to_metres(frame_number)

            array = depth_in_metres_calibrator(D_in_metres, kp, params['debug'])
            P3D = array

            P3D_pc = o3d.geometry.PointCloud()
            P3D_pc.points = o3d.utility.Vector3dVector(P3D)
            # o3d.io.write_point_cloud(processed_path, P3D_pc)

            pcd_array = np.asarray(P3D_pc.points)

            if params['debug']:
                draw_point_cloud(P3D_pc)

            # isolating face from torso.................................................
            pcd_foreground_face = pcd_array[pcd_array[:, 1] > face_thresh, :]

            # rotate to compensate small pitch rot
            pcd_foreground_face = rotate_ndarray_3d(pcd_foreground_face, rot_angle, [1, 0, 0])

            P3D_pc_fore_f = o3d.geometry.PointCloud()
            P3D_pc_fore_f.points = o3d.utility.Vector3dVector(pcd_foreground_face)

            if params['debug']:
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

            lm_nose_tip_prova1 = find_nose_tip_with_landmark(pcd_foreground_face)
            # .....................................................

            pcd_f_face = o3d.geometry.PointCloud()
            pcd_f_face.points = o3d.utility.Vector3dVector(pcd_foreground_face)
            pcd_f_face_array = np.asarray(pcd_f_face.points)

            o3d.io.write_point_cloud(savepath, pcd_f_face)

            if params['debug']:
                draw_point_cloud_with_landmarks(pcd_f_face, lm_nose_tip_prova1, 'Lm post crop')
else:
    # SCAN HD ------------------------------------------------------------
    # take the .obj point cloud in meshlab and save it as .ply and load it

    params = {'debug': True, 'writePc': False}

    HR_path = '/mnt/storage2-data-2T/LR-HR-MICC-dataset/HR/withobj/'

    subj_list = glob.glob(HR_path + '*')

    rot_angle = -25
    radius = 105

    for s in subj_list:
        models_list = glob.glob(s + '/*[!.txt]')  # exclude txt
        print('Doing subject {}'.format(s))
        for m in models_list:

            print('Doing Model {}'.format(m))
            model_path = glob.glob(m + '/*.wrl')
            model_name = model_path[0].split('/')[-1][:-4]

            processed_path = '/'.join(model_path[0].split('/')[:-4]) + '/processed/' + model_path[0].split('/')[-3]

            if os.path.isfile(processed_path + '/' + model_name + '_cropped.ply'):
                continue

            if not os.path.exists(processed_path):
                os.mkdir(processed_path)

            objfile = pywavefront.Wavefront(model_path[0], create_materials=True)  # collect_faces
            # start_pc_rgb = o3d.io.read_point_cloud("3D point clouds/" + hd_scan_number + ".ply")
            start_pc_rgb_array = np.asarray(objfile.vertices)
            # faces = np.asarray(objfile.mesh_list[0].faces)

            start_pc_rgb = o3d.geometry.PointCloud()
            start_pc_rgb.points = o3d.utility.Vector3dVector(start_pc_rgb_array)

            if params['debug']:
                draw_point_cloud(start_pc_rgb)

            # Rotate to make it front-facing
            start_pc_rgb_array = rotate_ndarray_3d(start_pc_rgb_array, rot_angle, [1, 0, 0])

            start_pc_rgb = o3d.geometry.PointCloud()
            start_pc_rgb.points = o3d.utility.Vector3dVector(start_pc_rgb_array)
            # Save rotated mesh
            o3d.io.write_point_cloud(processed_path + '/' + model_name + '_rotated.ply', start_pc_rgb)

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
            o3d.io.write_point_cloud(processed_path + '/' + model_name + '_cropped.ply', start_pc_rgb)

            if params['debug']:
                draw_point_cloud(start_pc_rgb)

# cloud = pv.PolyData(pcd_foreground_face_rgb)
# alpha_delauney = 2.2
# volume = cloud.delaunay_3d(alpha=alpha_delauney)
# shell = volume.extract_geometry()
# shell.plot()

# on depth
# depth_threshold = 20
# defShape_K, modGT_K = fitting_3dMM(pcd_f_face, depth_threshold, **params)
#
# # on scan
# rgb_threshold = 38
# defShape_HD, modGT_HD = fitting_3dMM(start_pc_rgb, rgb_threshold, **params)
#
# dist_from_kinect_and_HD = np.mean(cdist(defShape_K, defShape_HD))
#
# print('Distance from defShape of Kinect and HD: ' + str(dist_from_kinect_and_HD))
