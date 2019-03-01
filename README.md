# Student fork of SnapNet: point cloud semantization using deep segmentation network
By Raphaël Ginoulhac (Ecole des Ponts et Chaussées, Paris, 2018).

## Introduction
This project is a student fork of SnapNet, "Unstructured point cloud semantic labeling using deep segmentation networks", A.Boulch and B. Le Saux and N. Audebert, Eurographics Workshop on 3D Object Retrieval 2017
You can refer to the original paper and code (https://github.com/aboulch/snapnet) for details (Abstract and paper [here](https://aboulch.github.io/publication/2017-3DOR-snapnet).

This fork focused on semantic segmentation, with the goal of comparing three datasets : Scannet, Semantic-8 and Bertrand Le Saux aerial LIDAR dataset.
To achieve that, we clean, document, refactor, and improve the original project. 
We will compare the same datasets later with PointNet2, another state-of-the-art semantic segmentation project.

## Dependancies and data
We work on Ubuntu 16.04 with 3 GTX Titan Black and a GTX Titan X. On older GPUs, like my GTX 960m, you can expect to lower the number of points and the batch size for the training, otherwise you will get a OutOfMemory from TensorFlow.
You have to install TensorFlow on GPU (we use TF 1.2, cuda 8.0, python 2.7, but it should also work on newer versions with minor changes). You may have to install some additionnal Python modules.

Get the preprocessed data :
- Scannet : https://onedrive.live.com/?authkey=%21AHEO5Ik8Hn4Ue2Y&cid=423FEBB4168FD396&id=423FEBB4168FD396%21136&parId=423FEBB4168FD396%21134&action=locate
- Semantic : http://www.semantic3d.net/view_dbase.php?chl=1

### Citing the original GitHub : 

#### License

This code and weights are both released under a dual license, research and commercial.
Basically the code is GPLv3 for open access (contact us for non open purposes) and the weights are released under Creative Commons BY-NC-SA (contact us for non open purposes).
See the [license](LICENSE.md).

## Code

The code is composed of two main parts: a C++ library and a python scripts. The C++ library permits to load the files at Semantic 3D format and compute the composite features as described in the paper.

The python scripts generates the views using the Pyqtgraph 3D viewer and predict the segmentation using TensorFlow.

### Dependencies

##### C++
- Cython
- PCL, for installation see [pointclouds.org](pointclouds.org).
- OpenMP

Dependencies in the pointcloud_tools repository : 
- NanoFlann: nanoflann.hpp should be included in the include directory
- Eigen: Eigen should also be included in the include directory
Please note that for Eigen and Nanoflann new versions, you must check the corresponding website.


#### Python
- TensorFlow: it is the deep learning framework used in this implementation
- TQDM, Scipy, Numpy ...

- (Not necessary on last update) Kaffe: if you want to train a new model using the VGG16 weights, they need to be converted to format compatible with TensorFlow.


In our implementation we used the caffe weights avalaible [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md).


### Building

To generate the C++/Python library.

    cd pointcloud_tools
    python setup.py install --home="."

It will build the library.

**Note**: in the ```setup.py```, we have filled the path with the standard path for PCL installation libraries and headers on Ubuntu 16.04. It is hardcoded, you need to modify the paths in setup.py if the libraries are elsewhere on your computer.

### Configuration file

The Configuration file is a json file containing the parameters and paths:

    {
        "train_input_dir":"path_to_directory_TRAIN",
        "test_input_dir":"path_to_directory_TEST",
        "train_results_root_dir":"where_to_put_training_products",
        "test_results_root_dir":"where_to_put_test_products",
        "images_dir":"images",

        training:true,

        "imsize":224,
        "voxel_size":0.1,

        "train_cam_number":10,
        "train_create_mesh" : true,
        "train_create_views" : true,
        "train_create_images" : true,

        "test_cam_number":10,
        "test_create_mesh" : true,
        "test_create_views" : true,
        "test_create_images" : true,

        "vgg_weight_init":"path_to_vgg_weights",
        "batch_size" : 24,
        "learning_rate" : 1e-4,
        "epoch_nbr" : 100,
        "label_nbr" : 10,
        "input_ch" : 3,

        "train_rgb" : true,
        "train_composite" : true,
        "train_fusion" : true,

        "saver_directory_rgb" : "path_to_rgb_model_directory",
        "saver_directory_composite" : "path_to_composite_model_directory",
        "saver_directory_fusion" : "path_to_fusion_model_directory",
        "output_directory":"path_to_output_product_directory"
    }

It is used in the python scripts, to avoid code recopy.


### Launching the python scripts

Firstly, the name of the input files are hardcoded in some python files : sem3_gen_images.py (both train and test set), sem3d_test_backproj.py and sem3d_test_to_sem3D_labels.py (test set for those two). If you want to adapt Snapnet to another dataset, you should modify those files. Consider that the load functions are coded in C++ and it may be hard to modify them, so you should transform your dataset in filename.txt and filename.labels ; and then hardcode those filenames. The rest of the configuration is self-explanatory in the config.json file.

For the training and testing dataset, the point cloud decimation, views and images generation are called with:

    python3 sem3d_gen_images.py --config config.json
    
If you're getting an error like : 
    Not enough neighbors are considered: ffn or sfn out of range! Consider increasing nnn... Setting R=426 to be BOUNDA RY!
    Not enough neighbors are considered: source of R=792 is out of range! Consider increasing nnn...
    Number of neighborhood size increase requests for fringe neighbors: 2117
    Number of neighborhood size increase requests for source: 278
    
go in both PointCloud.cxx and PointCloudLabels.cxx that are in pointcloud_tools/, and change 
    gp3.setMaximumNearestNeighbors (300);
with the value you need. Then execute setup.py to compile the C++ code.
    
You just have to specify training=true/false in config.json. It is necessary to create the mesh, views and images for the testing phase.

Further more, you need a display to generate the images with Qt, so if you're using a SSH connection like me, you should execute :
    DISPLAY:=0 python3 sem3d_gen_images.py --config config.json
This will redirect the display and the program will work.

To train the models (rgb, composite and fusion) from scratch, run:

    python3 sem3d_train_tf.py --config config.json
 
This will take 3*number_of_epochs (100 by default), and is also necessary for the testing phase.

The semantic predictions on images and back-projection on the decimated clouds can be called using:

    python3 sem3d_test_backproj.py --config config.json

Finally to generate the files at the Semantic 3D format and assign a label to each point of the original point cloud, run;

    python3 sem3d_test_to_sem3D_labels.py --config config.json

## Initial weights

You should download the weights used here : https://gist.github.com/ksimonyan/211839e770f7b538e2d8
They are called with "vgg_weights" in the code.

##Information

name: 16-layer model from the arXiv paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"

caffemodel: VGG_ILSVRC_16_layers

caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel

license: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

caffe_version: trained using a custom Caffe-based framework

gist_id: 211839e770f7b538e2d8
