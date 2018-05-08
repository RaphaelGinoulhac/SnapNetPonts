# SnapNet for semantic segmentation of 3d points clouds
By Raphaël Ginoulhac (Ecole des Ponts et Chaussées, Paris, 2018).

## Introduction
This project is a student fork of SnapNet, "Unstructured point cloud semantic labeling using deep segmentation networks", A.Boulch and B. Le Saux and N. Audebert, Eurographics Workshop on 3D Object Retrieval 2017
You can refer to the original paper and code (https://github.com/aboulch/snapnet) for details.

This fork focused on semantic segmentation, with the goal of comparing three datasets : Scannet, Semantic-8 and Bertrand Le Saux aerial LIDAR dataset.
To achieve that, we clean, document, refactor, and improve the original project. 
We will compare the same datasets later with PointNet2, another state-of-the-art semantic segmentation project.

## Dependancies and data
We work on Ubuntu 16.04 with 3 GTX Titan Black and a GTX Titan X. On older GPUs, like my GTX 960m, you can expect to lower the number of points and the batch size for the training, otherwise you will get a OutOfMemory from TensorFlow.
You have to install TensorFlow on GPU (we use TF 1.2, cuda 8.0, python 2.7, but it should also work on newer versions with minor changes). You may have to install some additionnal Python modules.

Get the preprocessed data :
- Scannet : https://onedrive.live.com/?authkey=%21AHEO5Ik8Hn4Ue2Y&cid=423FEBB4168FD396&id=423FEBB4168FD396%21136&parId=423FEBB4168FD396%21134&action=locate
- Semantic : ask us

## Roadmap
