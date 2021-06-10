import scipy.io
from scipy.spatial.distance import cdist
import sys
import _3DMM
import Matrix_operations as mo
import mat73
import numpy.matlib as npm
from utils import *


def bidirectionalAssociation(modGT, defShape):
    D = cdist(modGT, defShape)
    mindists = np.amin(D, axis=0)
    minidx = np.argmin(D, axis=0)
    mindistsGT = np.amin(D, axis=1)
    minidxGT = np.argmin(D, axis=1)
    threshGlobal = np.mean(mindistsGT) + np.std(mindistsGT)
    toRemGlobal = mindistsGT > threshGlobal
    idxGlobal = np.where(toRemGlobal)[0]
    unGT = np.unique(minidxGT)

    modPerm = np.zeros(np.shape(defShape))
    for i in range(len(unGT)):
        kk = np.where(minidxGT == unGT[i])
        kk = np.asarray(kk)
        thresh = np.mean(mindistsGT[kk]) + np.std(mindistsGT[kk])
        toRem = kk[mindistsGT[kk] > thresh]
        toRem = np.concatenate((toRem, idxGlobal))
        kk = np.setdiff1d(kk, toRem)
        if len(kk) > 1:
            modPerm[unGT[i], :] = np.mean(modGT[kk, :], axis=0)
        elif len(kk) == 1:
            modPerm[unGT[i], :] = modGT[kk, :]

    missed = np.nonzero(np.sum(modPerm, 1) == 0)
    missed = np.asarray(missed)
    modPerm[missed, :] = modGT[minidx[missed], :]

    err = np.mean(mindists)
    return modPerm, err, minidx, missed


def reassociateDuplicates(modGT, defShape):
    D = cdist(modGT, defShape)
    # Case for average model
    if np.size(modGT) == np.size(defShape):
        D = D + np.identity(np.size(D)) * sys.float_info.max
    mindists = np.amin(D, axis=0)
    minidx = np.argmin(D, axis=0)
    [un, iidx] = np.unique(minidx, True)

    iidxMiss = np.setdiff1d(np.arange(0, np.size(defShape, 0)), iidx)
    df = defShape[iidxMiss, :]

    modTmp = modGT
    modPerm = np.zeros((np.size(defShape, axis=0), 3))
    modPerm[iidx, :] = modTmp[un, :]

    while np.size(df) != 0:
        # Remove unique vertices from gt
        modTmp = np.delete(modTmp, un, 0)
        # Re-compute distances
        D = cdist(modTmp, df)
        minidx = np.argmin(D, axis=0)
        [un, iidx] = np.unique(minidx, True)

        # Store new unique indices and remove from the old unique set
        iidx_n = iidxMiss[iidx]
        iidxMiss = np.delete(iidxMiss, iidx)
        df = np.delete(df, iidx, 0)

        # Add to the new model
        modPerm[iidx_n, :] = modTmp[un, :]

    err = np.mean(mindists)
    return modPerm, err


def fitting_3dMM(source, threshold, **kwargs):
    mat_op = mo.Matrix_op
    _3DMM1 = _3DMM._3DMM()

    # Optimization params
    derr = 0.01
    maxIter = 30
    lambda_all = 1

    # Load 3DMM Components
    slc = mat73.loadmat(kwargs['datapath'] + '/SLC_300_1_0.1.mat')
    components = slc.get('Components')
    weights = slc.get('Weights')
    aligned_models_data = None
    components_R = mat_op(components, aligned_models_data)
    Components_res = components_R.X_res

    # Load avgModel
    avgM = mat73.loadmat(kwargs['datapath'] + '/avgModel_bh_1779_NE.mat')
    avgModel = avgM.get('avgModel')

    # Zero mean
    baric_avg = np.mean(avgModel, axis=0)
    avgModel = avgModel - npm.repmat(baric_avg, np.size(avgModel, axis=0), 1)

    # Create deformed shape struct to be updated
    defShape = avgModel
    dShape = o3d.geometry.PointCloud()
    dShape.points = o3d.utility.Vector3dVector(defShape)

    vertex = source.points
    vertex = np.asarray(vertex)

    # calculate top and bottom point of avgModel and newModel for scale.
    avg_landmark_top_face_value = np.max(avgModel[:, 1])
    avg_landmark_bottom_face_value = np.min(avgModel[:, 1])

    depth_landmark_top_face_value = np.max(vertex[:, 1])
    depth_landmark_bottom_face_value = np.min(vertex[:, 1])

    scale_avg = avg_landmark_top_face_value - avg_landmark_bottom_face_value
    scale_depth = depth_landmark_top_face_value - depth_landmark_bottom_face_value
    scale_factor = np.divide(scale_avg, scale_depth)

    vertex = np.multiply(vertex, scale_factor)

    if np.size(vertex, axis=0) < np.size(avgModel, axis=0):
        print('Target model with less vertices than 3DMM!')

    # Zero mean GT model
    baric = np.mean(vertex, axis=0)
    modGT = vertex - npm.repmat(baric, np.size(vertex, axis=0), 1)

    # Load Control Landmarks
    frgcLm_buLips_gen = mat73.loadmat(kwargs['datapath'] + '/landmarksFRGC_CVPR20_ver2.mat')
    frgcLm_buLips = frgcLm_buLips_gen.get('frgcLm_buLips')
    # Make 0-based indexing
    lm3dmmGT = frgcLm_buLips - 1

    # Find nose-tip and translate to make nose-tips coincident
    nt = np.where(modGT[:, 2] == np.max(modGT[:, 2]))
    nt = nt[0]
    ntTrasl = avgModel[int(lm3dmmGT[5]), :] - modGT[int(nt[0]), :]
    modGT = modGT + ntTrasl

    # Initial ICP
    mGT = o3d.geometry.PointCloud()
    mGT.points = o3d.utility.Vector3dVector(modGT)

    if kwargs['debug']:
        draw_point_cloud_with_landmarks(mGT, dShape, 'Before ICP registration')

    # Perform ICP
    icp_result = o3d.pipelines.registration.registration_icp(mGT, dShape, threshold)

    transf_vec = np.asarray(icp_result.transformation)
    Ricp = transf_vec[0:3, 0:3]
    Ticp = transf_vec[0:3, 3]

    modGT = np.transpose(np.matmul(Ricp, modGT.T) + np.transpose(npm.repmat(Ticp, np.size(modGT, axis=0), 1)))
    mGT1 = o3d.geometry.PointCloud()
    mGT1.points = o3d.utility.Vector3dVector(modGT)
    if kwargs['debug']:
        draw_point_cloud_with_landmarks(mGT1, dShape, 'Post ICP registration')

    o3d.io.write_point_cloud(kwargs['savepath'] + "/modGT.ply", mGT1)
    # Initial Association
    [modPerm, err, minidx, missed] = bidirectionalAssociation(modGT, defShape)

    # Re-align
    iidx = np.setdiff1d(np.arange(0, np.size(defShape, 0)), missed)
    [A, S, R, trasl] = _3DMM1.estimate_pose(modGT[minidx[iidx], :], defShape[iidx, :])
    modPerm = np.transpose(_3DMM1.getProjectedVertex(modPerm, S, R, trasl))
    modGT = np.transpose(_3DMM1.getProjectedVertex(modGT, S, R, trasl))
    print('Mean distance initialization: ' + str(err))

    print('Start NRF routine')
    d = 1
    t = 1
    while t < maxIter and d > derr:
        # Fit the 3dmm
        alpha = _3DMM1.alphaEstimation_fast_3D(defShape, modPerm, Components_res, np.arange(0, 6704), weights,
                                               lambda_all)
        defShape = np.transpose(_3DMM1.deform_3D_shape_fast(np.transpose(defShape), components, alpha))

        # Re-associate points as average
        [modPerm, errIter, minidx, missed] = bidirectionalAssociation(modGT, defShape)
        d = np.abs(err - errIter)
        err = errIter
        print('Mean distance: ' + str(err))

        # Iterate
        t = t + 1

    # Debug...................................

    landmarks_defShape = defShape[lm3dmmGT.astype(int), :]
    landmarksdShape = o3d.geometry.PointCloud()
    landmarksdShape.points = o3d.utility.Vector3dVector(landmarks_defShape)

    dShape = o3d.geometry.PointCloud()
    dShape.points = o3d.utility.Vector3dVector(defShape)

    o3d.io.write_point_cloud(kwargs['savepath'] + "/defShape.ply", dShape)

    if kwargs['debug']:
        print('Landmarks in red and defShape in blue')
        draw_point_cloud_with_landmarks(dShape, landmarksdShape, 'Figure')

    # Registered GT model building ..................
    print('Start Dense Registration routine')
    modFinal, err = reassociateDuplicates(modGT, defShape)
    print('Done!')
    # ................................................

    mFinal = o3d.geometry.PointCloud()
    mFinal.points = o3d.utility.Vector3dVector(modFinal)
    o3d.io.write_point_cloud(kwargs['savepath'] + "/mod_Final.ply", mFinal)
    if kwargs['debug']:
        print('landmarks in red and modFinal in blue')
        draw_point_cloud_with_landmarks(mFinal, landmarksdShape, 'Figure')

    return defShape, modGT