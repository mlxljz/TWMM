
import cv2
import os
import numpy as np
import random

from .image_process import center_crop,img_read,homo_save,get_cores_point,show_map_and_img,mosaic_img,stitch_img
from .tools import exist_or_mkdir,list_to_csv
# from template_based.CFOG import CFOG_descriptor,cfog_matching_points
# from TAMM_clean.TAMM import TAMM
# from deep_learning.main_RANSAC import resnet_feature
from feature_based.sift_flow_torch import SIFT_feature

def draw_match_point(thermal_point_list,sen_point_list,outliers_ref,outliers_sen,thermal_color,visible_img):
    #正确的点用白色
    thermal_color=thermal_color.copy()
    visible_img=visible_img.copy()
    for thermal_point,sen_point in zip(thermal_point_list,sen_point_list):
        thermal_point = int(thermal_point[0]),int(thermal_point[1])
        sen_point = int(sen_point[0]), int(sen_point[1])
        cv2.circle(thermal_color,(thermal_point[1],thermal_point[0]),3,(255,255,255),-1)
        cv2.circle(visible_img,(sen_point[1],sen_point[0]),3,(255,255,255),-1)
    # 错误的点用黑色

    for thermal_point,sen_point in zip(outliers_ref,outliers_sen):
        cv2.circle(thermal_color,(thermal_point[1],thermal_point[0]),3,(0,0,0),-1)
        cv2.circle(visible_img,(sen_point[1],sen_point[0]),3,(0,0,0),-1)
    return thermal_color, visible_img

def draw_match_line(thermal_point_list,sen_point_list,outliers_ref,outliers_sen,thermal_color,visible_img):
    stitched_thermal_visible = stitch_img([thermal_color, visible_img], hor_flag=1,   spacing=0)


    imgsize = thermal_color.shape[1]
    for point_t,point_v in zip(thermal_point_list,sen_point_list):
        point_t = int(point_t[0]),int(point_t[1])
        point_v = int(point_v[0]),int(point_v[1])
        color = int(255*random.random()),int(255*random.random()),int(255*random.random())
        cv2.circle(stitched_thermal_visible,(point_t[1], point_t[0]),5,color)
        cv2.circle(stitched_thermal_visible, (point_v[1] + imgsize, point_v[0]), 5, color)
        cv2.line(stitched_thermal_visible, (point_t[1], point_t[0]), (point_v[1] + imgsize, point_v[0]),
                 color, 2)
    return stitched_thermal_visible

def draw_point(thermal_point_list,sen_point_list,outliers_ref,outliers_sen,thermal_color,visible_img):
    #如果匹配点个数大于50，则画点方式
    if len(thermal_point_list)>50:
        match_point_img = draw_match_point(thermal_point_list, sen_point_list, outliers_ref, outliers_sen, thermal_color, visible_img)
    else:
        #如果匹配点个数小于50，则画线方式
        match_point_img =draw_match_line(thermal_point_list,sen_point_list,outliers_ref,outliers_sen,thermal_color,visible_img)
    return match_point_img

class ThermalVisble(object):
    def __init__(self,
                 thermal_tiff_path,
                 thermal_rgb_path,
                 visible_rgb_path,
                 scale,
                 crop_size,
                 attention_flag=False
                 ):
        thermal_tiff=img_read(thermal_tiff_path,scale['thermal']) if scale is not None else img_read(thermal_tiff_path,2.3)
        thermal_rgb=img_read(thermal_rgb_path,scale['thermal']) if scale is not None else img_read(thermal_tiff_path,2.3)
        visible_rgb=img_read(visible_rgb_path,1)

        self.img_name = os.path.split(thermal_tiff_path)[1].split('.')[0]
        self.thermal_tiff=center_crop(thermal_tiff,crop_size)
        self.thermal_rgb=center_crop(thermal_rgb,crop_size)
        self.visible_rgb=center_crop(visible_rgb,crop_size)
        self.visible_rgb_gray=cv2.cvtColor(self.visible_rgb,cv2.COLOR_BGR2GRAY)
        self.attention_thermal=None

    def get_img_features(self,**kwargs):
        method = kwargs['method']
        if method=='CFOG':
            cfog_thermal_tiff = CFOG_descriptor(img=self.thermal_tiff.copy(),bin_size=kwargs['bin_size'])
            self.thermal_fea,self.fea_mag, self.thermal_fea_img,self.thermal_fea_img_norm = cfog_thermal_tiff.extract()
            cfog_visible = CFOG_descriptor(img=self.visible_rgb_gray.copy(),bin_size=kwargs['bin_size'])
            self.visible_fea,_ ,self.visible_fea_img,self.thermal_fea_img_norm = cfog_visible.extract()
            return  self.thermal_fea,self.visible_fea
        elif method == 'SIFT':
            self.thermal_fea = SIFT_feature(self.thermal_tiff.copy())[0]
            self.visible_fea = SIFT_feature(self.visible_rgb_gray.copy())[0]
            return self.thermal_fea,self.visible_fea

        elif method=='resnet':
            self.thermal_fea = resnet_feature(self.thermal_rgb.copy())
            self.visible_fea = resnet_feature(self.visible_rgb.copy())
            return self.thermal_fea, self.visible_fea
        else:
            return None

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
            return self.corres_points
        elif method == 'CFOG':
            patch_size=kwargs['patch_size']
            search_radius=kwargs['search_radius']
            num_points=kwargs['num_points']
            corres_points=cfog_matching_points(ref_img=self.thermal_tiff,
                                 ref_fea=self.thermal_fea,
                                 sen_fea=self.visible_fea,
                                 attention=self.attention_thermal,
                                 window_size=patch_size,
                                 search_rad=search_radius,
                                 num_points=num_points)

            self.corres_points=corres_points
            return corres_points
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
        #在图像中写入homography的值
        # for index in range(3):
        #     warp_thermal=cv2.putText(warp_thermal,"%.2f %.2f %.2f"%(homo[index][0],homo[index][1],homo[index][2]),
        #                              (100,40+40*index),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
        self.warp_thermal=warp_thermal
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
            if key == 'mosaic':
                img_A =  getattr(self,'visible_rgb')
                img_B = getattr(self, 'warp_thermal')
                mosaic_AB = mosaic_img(img_A,img_B)
                # mosaic_BA = mosaic_img(img_B, img_A)
                cv2.imwrite(save_path, mosaic_AB)

            if key == 'matchpoints_img':
                save_value = getattr(self, key)
                thermal_path = os.path.join(root_path, key, self.img_name + '.' + 'jpg')
                visible_path = os.path.join(root_path, key, self.img_name + '.' + 'png')
                if isinstance(save_value,tuple):
                    cv2.imwrite(thermal_path,save_value[0])
                    cv2.imwrite(visible_path, save_value[1])
                else:
                    thermal_path = os.path.join(root_path, key, self.img_name + '.' + 'jpg')
                    cv2.imwrite(thermal_path, save_value)
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

