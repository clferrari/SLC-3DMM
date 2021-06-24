# A Sparse and Locally Coherent Morphable Face Model (TPAMI 2021)

![alt text](https://github.com/clferrari/SLC-3DMM/blob/master/images/slc.png)

This repository contains the official Python implementation of SLC and Non-Rigid-Fitting algorithm (NRF). We present a fully automatic approach that leverages a 3DMM to transfer its dense semantic annotation across raw 3D faces, establishing a dense correspondence between them. We propose a novel formulation to learn a set of sparse deformation components with local support on the face that, together with an original non-rigid deformation algorithm, allow the 3DMM to precisely fit unseen faces and transfer its semantic annotation. Our solution builds upon the observation that the musculoskeletal structure induces neighboring vertices to move according to consistent patterns (principle of local consistency of motion), making it possible to approximate the movement of a local region with single motion vectors. We leverage this property by first learning a corpus of primary deformation directions from the aligned training scans. Then, we learn how to expand each primary direction to a localized set of vertices.

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