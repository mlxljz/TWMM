"""
配准成功率计算：
输入实际的homo,gt homo,点；然后计算点的误差（我们平均选100个点）
"""

import numpy as np
from common.image_process import get_cores_point
from common.thermal_visible import ThermalVisble
from feature_based.SIFT_SURF_ORB_RIFT_SCB_HOPC import sift_registration,surf_registration,orb_registration,rift_registration,scb_registration,hardnet_registration,tfeat_registration
# from template_based.CFOG import CFOG_registration
# from TAMM_clean.TAMM import TAMM_registration

def vis_correspoint(homo,point_dict,threhold,imgpath):
    #遍历每个点，计算其误差
    CP_list=[]
    num_correct=0
    for point in point_dict:
        gt_point=get_cores_point(point[0],point[1],homo)
        cores_point=point_dict[point][0],point_dict[point][1]
        error_point=np.linalg.norm(np.array(gt_point)-np.array(cores_point))
        CP_list.append((point, gt_point, cores_point,error_point))
        if error_point<threhold:
            num_correct=num_correct+1
    return CP_list,num_correct
        #先都用黑白色的点，后面再调试
        # if np.linalg.norm(np.array(gt_point)-np.array(cores_point))>threhold:

def point_generate(imgsize,num_point):
    point_list=[]
    step=int(imgsize/num_point)
    start=int(step)
    for index1 in range(start,imgsize,step):
        for index2 in range(start,imgsize,step):
            point=(index1,index2)
            point_list.append(point)
    return point_list


def one_img_process(thermal_tiff_path,visible_rgb_path,thermal_rgb_path,method,save_root,cfg=None):
    # homo_save_path, error_save_path, point_save_path = save_root
    #先缩放和裁剪，得到thermal和visible
    thermal_visible_c = ThermalVisble(thermal_tiff_path=thermal_tiff_path,
                                      thermal_rgb_path=thermal_rgb_path,
                                      visible_rgb_path=visible_rgb_path,
                                      scale={'thermal':2*1.15,'visible':1},
                                      crop_size=1000,
                                      attention_flag=False)

    dict_method={'sift':sift_registration,
                 'surf':surf_registration,
                 'orb':orb_registration,
                 'scb':scb_registration,
                 'rift':rift_registration,
                 'hardnet':hardnet_registration,
                 'tfeat':tfeat_registration,
                 # 'hopc': hopc_registration,
                 # 'cfog':CFOG_registration,
                 # 'tamm':TAMM_registration
                 }
    #在使用该方法得到homography和对应点
    if method in  ['sift','surf','orb','scb','rift','hopc']:
        homo,correspoint=dict_method[method](thermal_visible_c.thermal_tiff,thermal_visible_c.visible_rgb_gray)
    elif method in ['hardnet','tfeat']:
        homo, correspoint = dict_method[method](thermal_visible_c.thermal_rgb, thermal_visible_c.visible_rgb)
    elif method in ['cfog','tamm']:
        homo, correspoint = dict_method[method](thermal_visible_c,cfg)

    if homo is None:
        homo=np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]], dtype=np.float64)
    if method=='scb':
        homo=np.array(homo).reshape(-1)
        homo=[[homo[0],homo[1],homo[2]],[homo[3],homo[4],homo[5]]]
        homo.append([0,0,1])
        homo=np.array(homo)

    print(thermal_tiff_path,method,homo)
    thermal_visible_c.homo_warp(homo)

    #保存，保存homo,保存对应点，保存可视化结果(缩放裁剪后的图像，配准后的图像)
    save_dict={
               'homo':'csv',
                'visible_rgb':'png',
                'thermal_rgb':'png',
                'warp_thermal':'png',
               'mosaic':'png'
               }
    if method in ['hardnet','tfeat','sift','surf','orb','rift','hopc']:
        thermal_visible_c.draw_matchpoints(correspoint[0], correspoint[1], [], [])
        save_dict['matchpoints_img']='jpg'
    if method in ['cfog','tamm']:
        thermal_visible_c.draw_matchpoints(correspoint[0], correspoint[1], correspoint[2], correspoint[3])
        save_dict['matchpoints_img'] = 'jpg'
    thermal_visible_c.result_save(save_root, save_dict)


def cfg_forcfog_tamm(method):
    patch_size = 40 if method=='tamm' else 100

    return {'bin_size':9,'patch_size':patch_size,'search_radius':45,'num_points':625,'thresh':2,'level_max':4,'fea':'CFOG'}


if __name__=='__main__':
    #输入图像(原始图像)，输入方法，结果保存地址，得到结果(homo和可视化结果保存，缩放裁剪后的图像，配准后的图像)
    method_list=['scb']#['tfeat','hardnet','orb','sift','surf','rift','scb']

    img_name_list = ['12-09-57-454']#'21-11-28-547'#'21-11-32-580',21-11-28-547,21-10-58-550,21-11-14-582,21-11-16-552,21-11-18-549,21-11-20-556,
    for img_name in img_name_list:
        thermal_path = f'test_img/{img_name}-radiometric.tiff'
        thermal_rgb_path = f'test_img/{img_name}-radiometric.jpg'
        visible_path = f'test_img/{img_name}-visible.jpg'


        for method in method_list:
            output_path = f'test_img/out/{method}/{img_name}'
            one_img_process(thermal_path,  visible_path,thermal_rgb_path,
                            method=method,
                            save_root = output_path,
                            cfg = cfg_forcfog_tamm(method))







