# Description
This is the official code for the TWMM. TWMM (**T**emplate matching with **W**eights, **M**ultilevel local max-pooling, and **M**ax index backtracking) is a  registration method for UAV thermal infrared and visible images taken by the camera with two sensors.

We have released part of the dataset and source code. The full datasets and source code will be released soon.

# Run
These implementation of methods (SIFT, SURF, ORB, RIFT, RCB, TFeat, HardNet) have been released. One can run these methodsas follows:

  `git clone https://github.com/mlxljz/TWMM.git`
  
  `python results_SIFT_SURF_RIFT_SCB_HOPC.py`
  
  The implementation for RIFT and SCB refers to http://www.ivlab.org/publications.html and https://ljy-rs.github.io/web/
  
# Test imgs and Result
The experimental results fordifferent methods are located in this directory：test_img.

The experimental results fordifferent methods are located in this directory：test_img/out.

The subitems in experimental results: homo, matchpoints_img, mosaic, warp_thermal
  
## Contact

This repo is currently maintained by Lingxuan Meng (201811070106@std.uestc.edu.cn ).
