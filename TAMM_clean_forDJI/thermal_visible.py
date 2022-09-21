import rasterio
from rasterio.enums import Resampling
import cv2
import os
import numpy as np
import sys

from image_process import center_crop,img_read,homo_save,get_cores_point,show_map_and_img,mosaic_img
from tools import csv_to_list,exist_or_mkdir
from CFOG import CFOG_descriptor,cfog_matching_points
from TAMM import TAMM
from tools import list_to_csv

def draw_point(thermal_point_list,sen_point_list,outliers_ref,outliers_sen,thermal_color,visible_img):
    #正确的点用白色
    thermal_color=thermal_color.copy()
    visible_img=visible_img.copy()
    for thermal_point,sen_point in zip(thermal_point_list,sen_point_list):
        cv2.circle(thermal_color,(thermal_point[1],thermal_point[0]),3,(255,255,255),-1)
        cv2.circle(visible_img,(sen_point[1],sen_point[0]),3,(255,255,255),-1)
    # 错误的点用黑色
    for thermal_point,sen_point in zip(outliers_ref,outliers_sen):
        cv2.circle(thermal_color,(thermal_point[1],thermal_point[0]),3,(0,0,0),-1)
        cv2.circle(visible_img,(sen_point[1],sen_point[0]),3,(0,0,0),-1)
    return thermal_color,visible_img

class ThermalVisble(object):
    def __init__(self,
                 thermal_tiff_path,
                 thermal_rgb_path,
                 visible_rgb_path,
                 scale,
                 crop_size,
                 attention_flag=False
                 ):
        thermal_tiff=img_read(thermal_tiff_path,scale['thermal_upsample'])
        thermal_rgb=img_read(thermal_rgb_path,scale['thermal_upsample'])
        visible_rgb=img_read(visible_rgb_path,scale['visible_upsample'])

        self.img_name = os.path.split(thermal_tiff_path)[1].split('.')[0]
        self.thermal_tiff=center_crop(thermal_tiff,crop_size)
        self.thermal_rgb=center_crop(thermal_rgb,crop_size)
        self.visible_rgb=center_crop(visible_rgb,crop_size)
        self.visible_rgb_gray=cv2.cvtColor(self.visible_rgb,cv2.COLOR_BGR2GRAY)
        self.attention_thermal=None

    def get_img_features(self,method,**kwargs):
        if method=='CFOG':
            cfog_thermal_tiff = CFOG_descriptor(img=self.thermal_tiff.copy(),bin_size=kwargs['bin_size'])
            self.thermal_fea,self.fea_mag, self.thermal_fea_img,self.thermal_fea_img_norm = cfog_thermal_tiff.extract()
            cfog_visible = CFOG_descriptor(img=self.visible_rgb_gray.copy(),bin_size=kwargs['bin_size'])
            self.visible_fea,_ ,self.visible_fea_img,self.thermal_fea_img_norm = cfog_visible.extract()
            return  self.thermal_fea,self.visible_fea
    def get_correspoints(self,method,**kwargs):
        if method=='TAMM':
            self.MLRA_thermal_visible=TAMM(thermal_fea=self.thermal_fea,
                                           visible_fea=self.visible_fea,
                                           attention_thermal=self.attention_thermal,
                                           patch_size=kwargs['patch_size'],
                                           search_radius=kwargs['search_radius'])
            self.MLRA_thermal_visible.get_level0_dict()
            self.MLRA_thermal_visible.get_all_level_dict(kwargs['level_max'])
            self.MLRA_thermal_visible.get_entry_point()
            corres=self.MLRA_thermal_visible.get_corres_points()
            #corres中存储这{(A_x,A_y):[(B_x,B_y),score]
            self.corres_points=corres[0]
            return corres
        else:
            return
    def draw_matchpoints(self,thermal_point_list, sen_point_list, outliers_ref, outliers_sen):
        match_img=draw_point(thermal_point_list=thermal_point_list,
                   sen_point_list=sen_point_list,
                   outliers_ref=outliers_ref,
                   outliers_sen=outliers_sen,
                   thermal_color=self.thermal_rgb,
                   visible_img=self.visible_rgb)
        self.matchpoints_img=match_img
        self.len_good_points=len(thermal_point_list)

    def homo_warp(self,homo):
        #将热红外可视化影像做warp
        warp_thermal = cv2.warpPerspective(self.thermal_rgb, homo,
                                            (self.thermal_rgb.shape[1], self.thermal_rgb.shape[0]))
        warp_thermal_tiff = cv2.warpPerspective(self.thermal_tiff, homo,
                            (self.thermal_rgb.shape[1], self.thermal_rgb.shape[0]), borderValue=-999)
        #在图像中写入homography的值
        # for index in range(3):
        #     warp_thermal=cv2.putText(warp_thermal,"%.2f %.2f %.2f"%(homo[index][0],homo[index][1],homo[index][2]),
        #                              (100,40+40*index),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
        self.warp_thermal=warp_thermal
        self.warp_thermal_tiff = warp_thermal_tiff

        self.homo=homo
        return self.warp_thermal

    def result_save(self,root_path,save_dict):
        #指定父目录，然后指定要保存的内容和保存方式(使用字典的方式，key为要保存的内容，value为保存方式)
        exist_or_mkdir(root_path)
        for key in save_dict:
            exist_or_mkdir(os.path.join(root_path, key))
            save_path=os.path.join(root_path,key,self.img_name+'.'+str(save_dict[key]))

            if key in ['visible_rgb','thermal_rgb','warp_thermal']:
                save_value = getattr(self, key)
                cv2.imwrite(save_path,save_value)
            if key == 'warp_thermal_tiff':
                save_value = getattr(self, key)
                import tifffile as tiff
                tiff.imsave(save_path, save_value)
            if key == 'mosaic':
                img_A =  getattr(self,'visible_rgb')
                img_B = getattr(self, 'warp_thermal')
                mosaic_AB = mosaic_img(img_A,img_B)
                # mosaic_BA = mosaic_img(img_B, img_A)
                cv2.imwrite(save_path, mosaic_AB)

            if key == 'matchpoints_img':
                save_value = getattr(self, key)
                thermal_path=os.path.join(root_path,key,self.img_name+'.'+'jpg')
                visible_path=os.path.join(root_path,key,self.img_name+'.'+'png')
                cv2.imwrite(thermal_path,save_value[0])
                cv2.imwrite(visible_path, save_value[1])
            if key =='homo':
                save_value = getattr(self, key)
                homo_save(csv_path=save_path,H=save_value)
            if key=='corres_points':
                save_value = getattr(self, key)
                result=[]
                result.append(self.len_good_points)
                for _ in save_value:
                    A_x,A_y,B_x,B_y,score=_[0],_[1],save_value[_][0][0],save_value[_][0][1],save_value[_][1]
                    predict_point=get_cores_point(A_x,A_y,self.homo)
                    error=np.linalg.norm([B_x-predict_point[0],B_y-predict_point[1]])
                    result.append([A_x,A_y,B_x,B_y,score,error])
                #保存corres_points的坐标，得分，offset等
                list_to_csv(csvPath=save_path,listFile=result)
            if key=='attention_thermal':
                attention_rgb=show_map_and_img(map_img=self.attention_thermal,
                                             norm_flag='map',
                                             color_flag=False,
                                             cv_show_name=None,
                                             outpath=None)
                cv2.imwrite(save_path,attention_rgb)

