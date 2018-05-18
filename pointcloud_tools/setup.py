from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


pcl_include_dir = "/usr/include/pcl-1.7"
pcl_lib_dir = "/usr/local/lib"
vtk_include_dir = "/usr/include/vtk-6.2"
eigen_include_dir = "/usr/include/eigen3"
flann_include_dir = "/home/eleves/Documents/nanoflann"
numpy_include_dir = "/home/eleves/anaconda3/envs/tf/lib/python2.7/site-packages/numpy/core/include"

# depending on the distribution, change the directory according to the installed version
# example for stock ubuntu 16.04:
# vtk_include_dir = "/usr/include/vtk-5.10"

ext_modules = [Extension(
       "PcTools",
       sources=["Semantic3D.pyx", "Sem3D.cxx","pointCloud.cxx", "pointCloudLabels.cxx"],  # source file(s)
       include_dirs=["third_party_includes/",pcl_include_dir, vtk_include_dir, eigen_include_dir, flann_include_dir, numpy_include_dir],
       language="c++",             # generate C++ code
       library_dirs=[pcl_lib_dir],
       libraries=["pcl_common","pcl_kdtree","pcl_features","pcl_surface","pcl_io"],
       extra_compile_args = ["-fopenmp", "-std=c++11"]
  )]

setup(
    name = "PointCloud tools",
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
