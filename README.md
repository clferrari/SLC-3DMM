# A Sparse and Locally Coherent Morphable Face Model (TPAMI 2021)

This repository contains the official Python implementation of the method presented in

> **A Sparse and Locally Coherent Morphable Face Model for Dense Semantic Correspondence Across Heterogeneous 3D Faces** <br>
> Claudio Ferrari, Stefano Berretti, Pietro Pala, Alberto Del Bimbo <br>
> IEEE Transactions on Pattern Analysis and Machine Intelligence  

[[ArXiv](https://arxiv.org/abs/2006.03840)]

### Learning SLC-3DMM

To be updated.

### Data Pre-processing

To pre-process the data i.e. crop the face region, use preprocess_data.py. It works both for depth images e.g. Kinect and point-clouds. 

### Dense Registration Algorithm

To run the dense registration algorithm, run the fitting3DMM method in denseRegistration.py. 