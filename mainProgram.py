from denseRegistration import *
import pywavefront
import glob
import os

HD = True
params = {'debug': True, 'savepath': 'Fit-res', 'datapath': 'data'}

if not HD:
    start_pc = o3d.io.read_point_cloud('Ex-frames/depth_proc.ply')
    ICP_threshold = 20
    defShape_K, modGT_K = fitting_3dMM(start_pc, ICP_threshold, **params)
else:
    start_pc = o3d.io.read_point_cloud('Ex-pc/1_processed.ply')
    ICP_threshold = 38
    defShape_HD, modGT_HD = fitting_3dMM(start_pc, ICP_threshold, **params)

