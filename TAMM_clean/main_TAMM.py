import sys
import numpy as np
import cv2
import os
from thermal_visible import ThermalVisble
from image_process import filter_outliers,findHomoraphy
from tools import csv_to_list,filter_list
import random

multiprocess_flag=True








def one_img_process(therma_img_path,therma_rgb_path,visible_img_path,out_img,kwargs):


    thermal_visible_c = ThermalVisble(thermal_tiff_path=therma_img_path,
                                      thermal_rgb_path=therma_rgb_path,
                                      visible_rgb_path=visible_img_path,
                                      scale = kwargs['thermal_upsample'],
                                      crop_size=kwargs['crop_size'],
                                      attention_flag=False)
    import time
    s1=time.time()
    #提取CFOG特征
    thermal_visible_c.get_img_features(method='CFOG',bin_size=9)
    #CFOG TAMM配准
    corres=thermal_visible_c.get_correspoints(method='TAMM',
                                              patch_size=kwargs['patch_size'],
                                              search_radius=kwargs['search_radius'],
                                              level_max=kwargs['level_max'])
    s2 = time.time()
    print(s1-s2)
    thermal_visible_dict=corres[0]
    thermal_list=[]
    visible_list=[]
    for _ in thermal_visible_dict:
        thermal_list.append(_)
        visible_list.append(thermal_visible_dict[_][0])
    #filer points
    thermal_point_list, sen_point_list, outliers_ref, outliers_sen = filter_outliers(thermal_list, visible_list, thresh=2,method='NBCS')
    good = [[_thermal[1], _thermal[0], _sen[1], _sen[0]] for (_thermal, _sen) in zip(thermal_point_list, sen_point_list)]
    H_mat, flag_result= findHomoraphy(good)
    #做warp
    thermal_visible_c.homo_warp(H_mat)
    thermal_visible_c.draw_matchpoints(thermal_point_list, sen_point_list, outliers_ref, outliers_sen)
    #结果保存
    save_dict={'corres_points':'csv',
               'homo':'csv',
               'matchpoints_img':'jpg',
                'visible_rgb':'png',
                'thermal_rgb':'png',
                'warp_thermal':'png',
               'mosaic':'png'
               }

    thermal_visible_c.result_save(out_img,save_dict)
    return H_mat


if __name__=='__main__':
    global thermal_path, thermal_rgb_path, visible_path, output_path
    thermal_path = 'H:/temp/code/PFSegNets-master/imgs/input/11-44-43-767-radiometric.tiff'
    thermal_rgb_path = 'H:/temp/code/PFSegNets-master/imgs/input/11-44-43-767-radiometric.jpg'
    visible_path = 'H:/temp/code/PFSegNets-master/imgs/input/11-44-43-767-visible.jpg'
    output_path = 'H:/temp/code/PFSegNets-master/imgs/output/11-44-43-767'

    one_img_process(therma_img_path = thermal_path,
                    therma_rgb_path = thermal_rgb_path,
                    visible_img_path = visible_path,
                    out_img = output_path)