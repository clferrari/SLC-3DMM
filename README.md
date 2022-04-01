# A Sparse and Locally Coherent Morphable Face Model (TPAMI 2021)

![alt text](https://github.com/clferrari/SLC-3DMM/blob/master/images/slc.png)

This repository contains the official Python implementation of SLC and Non-Rigid-Fitting algorithm (NRF). We present a fully automatic approach that leverages a 3DMM to transfer its dense semantic annotation across raw 3D faces, establishing a dense correspondence between them. We propose a novel formulation to learn a set of sparse deformation components with local support on the face that, together with an original non-rigid deformation algorithm, allow the 3DMM to precisely fit unseen faces and transfer its semantic annotation. Our solution builds upon the observation that the musculoskeletal structure induces neighboring vertices to move according to consistent patterns (principle of local consistency of motion), making it possible to approximate the movement of a local region with single motion vectors. We leverage this property by first learning a corpus of primary deformation directions from the aligned training scans. Then, we learn how to expand each primary direction to a localized set of vertices.

> **A Sparse and Locally Coherent Morphable Face Model for Dense Semantic Correspondence Across Heterogeneous 3D Faces** <br>
> Claudio Ferrari, Stefano Berretti, Pietro Pala, Alberto Del Bimbo <br>
> IEEE Transactions on Pattern Analysis and Machine Intelligence  

[[ArXiv](https://arxiv.org/abs/2006.03840)]

## Changelog

06/24/21 - Repository created.

06/29/21 - Added HLS PCA components.

## Usage

### Requirements

The code is tested with Python3.8. Additional packages required are Open3D, pywavefront, NumPy, SciPy.

### Learning SLC-3DMM

To build the SLC-3DMM, we apply the method described in `Mairal, Julien, et al. "Online dictionary learning for sparse coding." ICML 2009`. Python and MATLAB packages are available at `http://spams-devel.gforge.inria.fr/downloads.html`.

In SLC folder, you can find mex files for the MATLAB implementation that we used. To learn the SLC components, you need to run `nmf_3dmm` MATLAB function.

`[Weights, Components] = nmf_3dmm(data,paramDL, paramLasso)`

`data` contains the training matrix of faces (same topology). The matrix needs to be in the form N x 3m where N is the number of training scans, and 3m are the linearized (x,y,z) coordinates of the vertices.

### Data Pre-processing

To pre-process the data i.e. crop the face region, use `preprocess_data.py`. It works both for depth images e.g. Kinect and point-clouds. 

### Dense Registration Algorithm

To run the dense registration algorithm, run the fitting3DMM method in `denseRegistration.py`. 

For a demonstration, run `mainProgram.py` 

### HLS-3DMM

We also release the PCA components (Matlab .mat file) learned on the 9,927 fully registered faces obtained with our method. They can be found at this link [[Components](https://drive.google.com/file/d/1CUeBHhtsRluw27tvhbmaiyR6n03bREmX/view?usp=sharing)], [[AverageModel](https://drive.google.com/file/d/1deJ-JE1AFWwTs84tIyidBzv7rk46zL-Y/view?usp=sharing)].

The average model includes the triangulation and the indices of the 68 landmarks. To deform the average model using the components, you can use the function `deform_3D_shape_fast` in `_3DMM.py`.

### Citation

If you find our work useful, cite us!

```
@ARTICLE{9462361,
  author={Ferrari, Claudio and Berretti, Stefano and Pala, Pietro and Del BIMBO, Alberto},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Sparse and Locally Coherent Morphable Face Model for Dense Semantic Correspondence Across Heterogeneous 3D Faces}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3090942}}
```

### License

The software is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.(see LICENSE).

### Contacts

For any inquiry, feel free to drop an email to ferrari.claudio88@gmail.com


