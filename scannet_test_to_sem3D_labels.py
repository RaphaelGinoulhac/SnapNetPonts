import numpy as np
import os
import scipy.misc
from tqdm import *
import json
#limit to a single GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from python.Semantic3D import Sem3D

# load the configuration file and define variables
print("Loading configuration file")
import argparse
parser = argparse.ArgumentParser(description='Semantic3D')
parser.add_argument('--config', type=str, default="config_scannet.json", metavar='N',
help='config file')
args = parser.parse_args()
json_data=open(args.config).read()
config = json.loads(json_data)

input_dir = config["test_input_dir"]
voxel_size = config["voxel_size"]
output_dir = config["output_directory"]

filenames = ["scene_test_" + str(j) for j in range(312)]

# create outpu directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in filenames:
    print(filename)

    semantizer = Sem3D()
    semantizer.set_voxel_size(voxel_size)

    mesh_filename = os.path.join(output_dir, filename+".ply")
    sem3d_cloud_txt = os.path.join(input_dir,filename+".txt")
    output_results = os.path.join(output_dir, filename+".txt")

    semantizer.mesh_to_label_file_no_labels(mesh_filename,sem3d_cloud_txt,output_results)
