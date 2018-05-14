#converts the scannet dataset from the pickle format to multiple .txt files
#each one representing a scene

import pickle
import os
import numpy as np

#hardcoding the path of pickle files
data_path="/home/eleves/Documents/projet_class_3D/data/scannet_data_pointnet2"
FILENAME_TRAIN = 'scannet_train.pickle'
FILENAME_TEST = 'scannet_test.pickle'

#extracting data
filenametrain = os.path.join(data_path, FILENAME_TRAIN)

with open(filenametrain, 'rb') as fp:
    scene_points_list_train = pickle.load(fp)
    semantic_labels_list_train = pickle.load(fp)

filenametest = os.path.join(data_path, FILENAME_TEST)

with open(filenametest, 'rb') as fp:
    scene_points_list_test = pickle.load(fp)
    semantic_labels_list_test = pickle.load(fp)


print(len(scene_points_list_train))
print(len(scene_points_list_test))

#for each scene, we save it to a .txt in the same folder as the pickle files
#there are 1201 train scenes! 
for i in range(len(scene_points_list_train)):
    np.savetxt(data_path+"/scene_train_"+ str(i)+".txt",scene_points_list_train[i])
    np.savetxt(data_path+"/labels_train_" + str(i)+".txt", semantic_labels_list_train[i])


for j in range(len(scene_points_list_test)):
    np.savetxt(data_path+"/scene_test_" + str(j)+".txt", scene_points_list_test[j])
    np.savetxt(data_path + "/labels_test_" + str(j) + ".txt", semantic_labels_list_test[j])