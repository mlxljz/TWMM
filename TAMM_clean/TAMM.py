import numpy as np
import torch
import cv2
import random
import os


from common.image_process import img_split,SSD_template_matching,show_map_and_img,filter_outliers,findHomoraphy
from common.tools import exist_or_mkdir

from joblib import Parallel, delayed
import multiprocessing
import itertools

def simi_xy(sour,tar,attention,patch_size,search_radius,multithread_flag=False):
    def one_patch_score(sour_xy_list,tar_dilate,attention,index_x,index_y,patch_size,search_radius):
        print(index_x,index_y)
        sour_xy = sour_xy_list[index_x][index_y]
        # cores为index_x,index_y对应的patch的中心位置(在tar img中)
        cores = index_x * patch_size + patch_size / 2, index_y * patch_size + patch_size / 2
        # cores为index_x,index_y对应的patch的中心位置(在tar_dilate img中)
        cores_dilate = cores[0] + search_radius, cores[1] + search_radius
        # cores_range为patch的左上角、右下角的位置
        cores_range = [int(cores_dilate[0] - (patch_size / 2) - search_radius),
                       int(cores_dilate[0] + (patch_size / 2) + search_radius),
                       int(cores_dilate[1] - (patch_size / 2) - search_radius),
                       int(cores_dilate[1] + (patch_size / 2) + search_radius)]
        tar_xy = tar_dilate[cores_range[0]:cores_range[1], cores_range[2]:cores_range[3]]

        attention_xy = None if attention is None else \
            attention[index_x * patch_size:(index_x + 1) * patch_size, index_y * patch_size:(index_y + 1) * patch_size]

        matchscore = SSD_template_matching(template=sour_xy, image=tar_xy, attention=attention_xy,
                                           return_score_flag=True)
        return matchscore

    sour_xy_list,num_index_x,num_index_y=img_split(sour, patch_size, overlap=0)
    # 为了处理好边缘，所以将tar做dilate,然后根据index_x和index_y取出对应的search_area
    tar_dilate = np.zeros([tar.shape[0] + 2 * search_radius, tar.shape[1] + 2 * search_radius, tar.shape[2]])
    tar_dilate[search_radius:-search_radius, search_radius:-search_radius] = tar
    if multithread_flag:
        #如果使用多线程计算,多线程目前有很大问题 8,6,12看看那个最快
        patch_score_xy_1=Parallel(n_jobs=16)(delayed(one_patch_score)(sour_xy_list,
                                           tar_dilate,
                                           attention,
                                           index_x,
                                           index_y,
                                           patch_size,
                                           search_radius)
                                          for index_x in range(num_index_x) for index_y in range(num_index_y))#itertools.product(range(num_index_x),range(num_index_y)))
        patch_score_xy=[]
        for index_x in range(num_index_x):
            patch_score_xy.append(patch_score_xy_1[index_x*num_index_y:index_x*num_index_y+num_index_y])

    else:
        patch_score_xy = []
        for index_x in range(num_index_x):
            patch_score_y = []
            for index_y in range(num_index_y):
                matchscore=one_patch_score(sour_xy_list=sour_xy_list,
                                           tar_dilate=tar_dilate,
                                           attention=attention,
                                           index_x=index_x,
                                           index_y=index_y,
                                           patch_size=patch_size,
                                           search_radius=search_radius)
                patch_score_y.append(matchscore)
            patch_score_xy.append(patch_score_y)
    return patch_score_xy

def res_next(dict_cur_level,dict_next_level,img_shape):
    cur_score_map=dict_cur_level['score_map']
    if not isinstance(cur_score_map,torch.Tensor):
        cur_score_map=torch.from_numpy(np.array(cur_score_map))

    # 其中dilation表示卷积核之间的距离
    max_pool=torch.nn.MaxPool2d(kernel_size=3, stride=1,dilation=1,padding=1,return_indices=True)
    cur_score_maxpool,cur_maxscore_indices=max_pool(cur_score_map)
    dict_cur_level['score_maxpool']=cur_score_maxpool
    cur_maxscore_indices=np.array(cur_maxscore_indices)
    cur_maxscore_indices_x=cur_maxscore_indices//cur_maxscore_indices.shape[3]
    cur_maxscore_indices_y=cur_maxscore_indices%cur_maxscore_indices.shape[3]
    cur_maxscore_indices=np.concatenate((cur_maxscore_indices_x[:,:,:,:,np.newaxis],cur_maxscore_indices_y[:,:,:,:,np.newaxis]),axis=4)
    dict_cur_level['max_indices']=cur_maxscore_indices

    cur_coor_list= dict_cur_level['coor_list']
    next_patch_size = dict_next_level['patch_size']
    next_start_coor = dict_next_level['start_coor']
    next_step_coor = dict_next_level['step']
    #不考虑边缘区域的patch,即每个patch都没有越过图像的边缘.
    next_coor_list = list(range(next_start_coor, img_shape[0] - next_start_coor + 1, next_step_coor))
    #从两个角度说明为什么没有考虑边缘：
    # (1)与实际情况不符合，可解释性差：patch score map的含义是：A图中以px,py为中心，p*p的patch，在允许一定变形情况下，在B图的(cx,cy)所能取得的相似度的最大值。
    # 如果我们考虑边缘的情况，则(px,py,p*p)的patch包含很多无效区域(例如，以左上角为中心的（px,py,p*p）其包含的有效区域只有整个区域的0.25)，这些区域的计算结果是不可信的，而且增加计算量。
    # next_coor_list = list(range(0, img_shape[0] + 1, next_step_coor))

    next_score_map=torch.zeros(len(next_coor_list),len(next_coor_list),
                               cur_score_maxpool.shape[2],cur_score_maxpool.shape[3])
    child_indices=np.zeros([len(next_coor_list),len(next_coor_list),4]).astype(np.int16)#用来记录每个当前patch对应的4个子patch的索引
    child_coor = np.zeros([len(next_coor_list), len(next_coor_list), 4]).astype(np.int16)
    #next_coor_x,next_coor_y表示patch的中心
    for next_coor_x in next_coor_list:
        for next_coor_y in next_coor_list:
            child_x0=cur_coor_list.index(next_coor_x-next_patch_size/4)
            child_x1=cur_coor_list.index(next_coor_x+next_patch_size/4)
            child_x=[child_x0,child_x1]
            child_y0 = cur_coor_list.index(next_coor_y - next_patch_size / 4)
            child_y1 = cur_coor_list.index(next_coor_y + next_patch_size / 4)
            child_y = [child_y0, child_y1]
            child_indices[next_coor_list.index(next_coor_x),next_coor_list.index(next_coor_y)]=[child_x[0],child_x[1],child_y[0],child_y[1]]
            child_coor[next_coor_list.index(next_coor_x),next_coor_list.index(next_coor_y)]=\
                [next_coor_x-next_patch_size/4,next_coor_x+next_patch_size/4,
                 next_coor_y - next_patch_size / 4,next_coor_y + next_patch_size / 4]
            next_score_xy_compose=cur_score_maxpool[child_x]
            next_score_xy_compose=next_score_xy_compose[:,child_y]
            next_score_xy=next_score_xy_compose.mean(dim=(0,1))
            next_score_map[next_coor_list.index(next_coor_x),next_coor_list.index(next_coor_y)]=next_score_xy

    dict_next_level['score_map']=next_score_map
    dict_next_level['child_indices'] = child_indices
    dict_next_level['child_coor'] = child_coor
    dict_next_level['coor_list'] = next_coor_list
    return dict_next_level


def parse_one_entry_point(dict_all_level_list,entry_point):
    """
    #得到dict_next_level_list后，从顶而下遍历的过程：
    #（1）先看当前位置是否有score，如果没有或当前得分更大，更新当前得分的score,否则则直接返回
    #(2) 对于它的每个子patch,获得对应的相似度(x，y)的位置；然后得到(x,y)周围3*3的区域的相似度得分哪个位置最大(simi_xb simi_yb)，最大得分为多少childscore
    #从而得到一组新的entry point: level patch_x patch_y simi_xb simi_yb score+childscore
    """
    level,px,py,cx,cy,score=entry_point
    if 'score_topdown_dict' not in dict_all_level_list[level]:
        dict_all_level_list[level]['score_topdown_dict']={}
    if level>0:
        if (px,py) not in dict_all_level_list[level]['score_topdown_dict'] or \
                score>dict_all_level_list[level]['score_topdown_dict'][(px,py)][2]:

            dict_all_level_list[level]['score_topdown_dict'].update({(px,py):[cx,cy,score]})
            #获得该patch对应的4个子patch
            child_indices=dict_all_level_list[level]['child_indices'][px,py]
            #依次计算子patch的对应，并继续向下迭代；否则，停止迭代，直接退出
            for child_x in [child_indices[0],child_indices[1]]:
                for child_y in [child_indices[2],child_indices[3]]:
                    cpx,cpy=child_x,child_y
                    ccx,ccy=cx,cy
                    ccx_withmaxoffset,ccy_withmaxoffset=dict_all_level_list[level-1]['max_indices'][cpx,cpy,ccx,ccy]
                    score_beforelevel=dict_all_level_list[level-1]['score_map'][cpx,cpy,ccx_withmaxoffset,ccy_withmaxoffset]
                    entry_point=level-1,cpx,cpy,ccx_withmaxoffset,ccy_withmaxoffset,score+score_beforelevel
                    parse_one_entry_point(dict_all_level_list, entry_point)
        else:
            return dict_all_level_list
    else:
        # level=0
        if (px, py) not in dict_all_level_list[level]['score_topdown_dict'] or \
                score > dict_all_level_list[level]['score_topdown_dict'][(px, py)][2]:
            dict_all_level_list[level]['score_topdown_dict'].update({(px, py): [cx, cy, score]})
    return dict_all_level_list

def parse_entry_points(dict_all_level_list,entry_point_list,search_radius):
    dict_all_level_list=dict_all_level_list.copy()
    for index,entry_point in enumerate(entry_point_list):
        dict_all_level_list = parse_one_entry_point(dict_all_level_list=dict_all_level_list,entry_point=entry_point)
    thermal_visible_dict = getpointlist(dict_all_level_list[0]['coor_list'], dict_all_level_list[0]['score_topdown_dict'],search_radius)
    return thermal_visible_dict,dict_all_level_list

def vis_entrypoint_parse_result(dict_all_level_list,imgA_o,imgB_o,search_radius=60,outpath=None):
    def draw_rect_cores(imgA,imgB,A_cor,B_cor,color):
        for img,cor in zip([imgA,imgB],[A_cor,B_cor]):
            cv2.rectangle(img,cor[0],cor[1],color,1)
        return imgA,imgB
    #遍历每一层
    index_top_level=len(dict_all_level_list) - 1
    for index in range(0,index_top_level+1):
        imgA,imgB=imgA_o.copy(),imgB_o.copy()
        dict_index_level=dict_all_level_list[index]
        score_topdown_dict=dict_index_level['score_topdown_dict']
        coor_list=dict_index_level['coor_list']
        patch_size=dict_index_level['patch_size']
        patchsize_half = int(patch_size / 2)
        for patch_cor in score_topdown_dict:
            px,py=patch_cor
            cx,cy,score=score_topdown_dict[patch_cor][0],score_topdown_dict[patch_cor][1],score_topdown_dict[patch_cor][2]
            #根据patch中心位置获得A图左上右下角坐标；根据patch中心和偏移获得B图左上右下角坐标
            A_center = np.array([coor_list[px],coor_list[py]])
            A_topleft, A_bottomright = A_center - patchsize_half, A_center + patchsize_half
            offset = np.array([cx - search_radius, cy - search_radius])
            B_topleft, B_bottomright = A_topleft + offset, A_bottomright + offset
            #cv2的先列后行，和array相反
            A_cor_cv2 = [(A_topleft[1], A_topleft[0]), (A_bottomright[1], A_bottomright[0])]
            B_cor_cv2 = [(B_topleft[1], B_topleft[0]), (B_bottomright[1], B_bottomright[0])]
            imgA, imgB = draw_rect_cores(imgA, imgB, A_cor_cv2, B_cor_cv2,
                                 color=(random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))
        if outpath:
            show_map_and_img(imgA,
                             norm_flag='colorimg',
                             color_flag=True,
                             cv_show_name=None,
                             outpath=os.path.join(outpath,'imgA'+'_'+str(index)+'.jpg'))
            show_map_and_img(imgB,
                             norm_flag='colorimg',
                             color_flag=True,
                             cv_show_name=None,
                             outpath=os.path.join(outpath,'imgB'+'_'+str(index)+'.jpg'))

def vis_entry_point(dict_all_level_list,entry_point_list,cv_show_name=None,outpath=None):
    #one_entry_point:level,px,py,cx,cy,score
    level_px_py_set=set([(_[0],_[1],_[2]) for _ in entry_point_list])
    score_map_img_list=[]
    for level_px_py in level_px_py_set:
        level,px,py=level_px_py[0],level_px_py[1],level_px_py[2]
        level_px_py_score_map=np.array(dict_all_level_list[level]['score_map'][px][py])
        score_map_img=show_map_and_img(map_img=level_px_py_score_map,
                                       norm_flag='map')
        level_px_py_points=[_ for _ in entry_point_list if _[0]==level and _[1]==px and _[2]==py]
        for point in level_px_py_points:
            cv2.circle(score_map_img,center=(point[4],point[3]),radius=2,color=(255,255,255),thickness=-1)
        score_map_img_list.append(score_map_img)
        if cv_show_name:
            cv2.imshow(cv_show_name+str(level_px_py),score_map_img)
            cv2.waitKey(1000)
        if outpath:
            cv2.imwrite(os.path.join(outpath,'entry_point_'+str(level)+str(px)+str(py)+'.jpg'),score_map_img)
    return score_map_img_list


def getpointlist(coor_list,corres_dict,search_radius):
    """
    coor_list:存储patch的中心坐标，如[40,80,120,160,....],通过coor_list将corres_dict中的索引转为图像中的坐标
    corres_dict:[px,py,cx,cy,score],可以根据coor_list[px],coor_list[py]得到A img中patch的中心坐标，
    根据coor_list[px]+cx-60,coor_list[py]+cy-60来获得B img中patch的中心坐标
    """
    thermal_list=[]
    visible_list=[]
    thermal_visible_dict={}
    for corres in corres_dict:
        # 遍历每一个可能的匹配，corres为A img patch的中心坐标，result为B img patch的中心坐标，然后存为两个列表
        cx,cy,score=corres_dict[corres][0],corres_dict[corres][1],corres_dict[corres][2].item()
        offset=[cx,cy]
        corres=(coor_list[corres[0]],coor_list[corres[1]])
        result=[c+r-search_radius for (c,r) in zip(corres,offset)]
        result=tuple(result)
        thermal_list.append(corres)
        visible_list.append(result)
        thermal_visible_dict.update({corres:(result,score)})
    return thermal_visible_dict

def vis_dict_level(dict_all_level_list,root_path,save_dict):
    exist_or_mkdir(root_path)
    for key in save_dict:
        if key=='score_map':
            # exist_or_mkdir(os.path.join(root_path, key))
            level=save_dict['score_map']['level']
            px = save_dict['score_map']['px']
            py = save_dict['score_map']['py']
            save_path = os.path.join(root_path, str(level) + '_' + str(px) + '_' + str(py) + '_scoremap' + '.jpg')
            level_pxpy_score=np.array(dict_all_level_list[level]['score_map'])[px][py]
            score_color = show_map_and_img(map_img=level_pxpy_score,
                                           norm_flag='map',
                                           color_flag=True,
                                           cv_show_name=None,
                                           outpath=save_path)
        if key=='entrypoint_parse':
            imgA_o=save_dict['entrypoint_parse']['imgA_o']
            imgB_o = save_dict['entrypoint_parse']['imgB_o']
            vis_entrypoint_parse_result(dict_all_level_list, imgA_o, imgB_o, search_radius=60,outpath=root_path)
        if key=='entrypoint':
            entry_point_list=save_dict['entrypoint']['point_list']
            # save_path = os.path.join(root_path, save_name)
            vis_entry_point(dict_all_level_list, entry_point_list, cv_show_name=None, outpath=root_path)




"""
elements in dict_level:
score_maxpool: dim(px,py,cx,cy) the maxpool value in (px,py,cx,cy)
max_indices: dim(px,py,cx,cy) the maxpool indices in (px,py,cx,cy)
coor_list: the center coordinate of patches
child_indices,child_coor:dim(px,py,4) indices or coordinate of patch(px,py)
"""
"by template matching with attention and multilevel maxpooling TAMM"
class TAMM(object):
    "定义MLRA的配准方法和相关的可视化方法以及一些结果保存方法"
    def __init__(self,
                 thermal_fea,
                 visible_fea,
                 attention_thermal,
                 patch_size,
                 search_radius):
        """
        thermal_fea:归一化后的特征
        visible_fea:归一化后的特征，thermal_fea.shape=visible_fea.shape
        TAMM采用模板匹配+多层聚合的方式
        patcnsize是我们认为的两张影像用位移配准后，偏差在亚像素的最小patch
        searchradius表明两张影像的偏差范围
        """
        self.thermal_fea=thermal_fea
        self.visible_fea=visible_fea
        self.attention_thermal=attention_thermal
        self.patch_size=patch_size
        self.half_patch_size=int(patch_size/2)
        self.search_radius=search_radius
        self.dict_all_level_list=[]
        self.img_shape=self.thermal_fea.shape[0],self.thermal_fea.shape[1]
        self.dict_all_level_list = []

    def get_level0_dict(self):
        """the score of level0 is obtained by template matching with attention"""
        # sour_patch_xy=img_split(self.thermal_fea, self.patch_size, overlap=0)
        level0_score_pxpycxcy=simi_xy(sour=self.thermal_fea,
                                      tar=self.visible_fea,
                                      attention=self.attention_thermal,
                                      patch_size=self.patch_size,
                                      search_radius=self.search_radius,
                                      multithread_flag=False)
        level0_score_pxpycxcy=np.array(level0_score_pxpycxcy)

        dict_cur_level={}
        dict_cur_level.update(
            {'level':0,
             'start_coor': self.half_patch_size,
             'step': self.patch_size,
             'score_map': level0_score_pxpycxcy,
             'patch_size': self.patch_size,
             'coor_list':list(range(self.half_patch_size,self.img_shape[0]-self.half_patch_size+1,self.patch_size))})
        self.dict_all_level_list.append(dict_cur_level)
        return self.dict_all_level_list,dict_cur_level
    def get_all_level_dict(self,level_max):
        for level_index in range(1, level_max):
            dict_next_level = {}
            dict_next_level.update({'level': level_index,
                                    'start_coor': self.patch_size * (2 ** (level_index - 1)),
                                    'step': self.patch_size,
                                    'patch_size': self.patch_size * (2 ** level_index)})

            dict_next_level = res_next(self.dict_all_level_list[level_index - 1], dict_next_level, img_shape=self.img_shape)
            self.dict_all_level_list.append(dict_next_level)
        return self.dict_all_level_list
    def get_pcxpy_entry_point(self,px,py):
        #默认entry point都是位于top level中
        index_top_level = len(self.dict_all_level_list) - 1
        one_entry_point_list = []
        for cx in range(0, self.dict_all_level_list[index_top_level]['score_map'].shape[2], 2):
            for cy in range(0, self.dict_all_level_list[index_top_level]['score_map'].shape[3], 2):
                entry_point = index_top_level, px, py, cx, cy, \
                              self.dict_all_level_list[index_top_level]['score_map'][px, py, cx, cy]
                one_entry_point_list.append(entry_point)
        one_entry_point_list.sort(key=lambda x: x[5], reverse=True)
        one_entry_point_list_top20=one_entry_point_list[0:int(len(one_entry_point_list) / 20)]
        return one_entry_point_list_top20

    def get_entry_point(self):
        self.entry_point_list=[]
        #将最高层level中的score_map前20%的像素结果看作entry_point
        index_top_level=len(self.dict_all_level_list) - 1
        for px in range(0, self.dict_all_level_list[index_top_level]['score_map'].shape[0],1):
            for py in range(0, self.dict_all_level_list[index_top_level]['score_map'].shape[1], 1):
                one_entry_point_list_top20=self.get_pcxpy_entry_point(px,py)
                self.entry_point_list.extend(one_entry_point_list_top20)
        self.entry_point_list.sort(key=lambda x: x[5], reverse=True)
        return self.entry_point_list

    def get_corres_points(self):
        thermal_visible_dict,self.dict_all_level_list = parse_entry_points(dict_all_level_list=self.dict_all_level_list,
                                                                                entry_point_list=self.entry_point_list,
                                                                                search_radius=self.search_radius)

        return thermal_visible_dict,self.dict_all_level_list

    def visiualize(self):
        #visiualize score map,params(level,px,py,outpath)
        level_pxpy_score = np.array(self.dict_all_level_list[level]['score_map'])[px][py]
        score_color=show_map_and_img(map_img=level_pxpy_score,
                                     norm_flag='map',
                                     color_flag=True,
                                     cv_show_name=None,
                                     outpath=outpath)

        #visiualize score_topdown_dict
        #默认已经得到了包含entrypoint处理后的dict_all_level_list，指定entrypoint,则从上倒下一一显示；指定某一层则显示某一层
        vis_entrypoint_parse_result(dict_all_level_list, imgA_o, imgB_o, search_radius=60)

        #visiualize entrypoint,输入dict_all_level_list，根据entry_point的px,py找到对应的patch，然后根据cx,cy绘制点
        vis_entry_point(dict_all_level_list, entry_point_list, cv_show_name=None, outpath=None)


def TAMM_registration(thermal_visible_c, cfg):
    # 提取CFOG特征
    thermal_visible_c.get_img_features(method=cfg['fea'],bin_size=cfg['bin_size'])
    # CFOG TAMM配准
    corres = thermal_visible_c.get_correspoints(method='TAMM',
                                                patch_size=cfg['patch_size'],
                                                search_radius=cfg['search_radius'],
                                                level_max=cfg['level_max'])
    thermal_visible_dict = corres
    thermal_list = []
    visible_list = []
    for _ in thermal_visible_dict:
        thermal_list.append(_)
        visible_list.append(thermal_visible_dict[_][0])
    # filer points
    thermal_point_list, sen_point_list, outliers_ref, outliers_sen = filter_outliers(thermal_list, visible_list,
                                                                                     thresh=cfg['thresh'],
                                                                                     method='NBCS')
    good = [[_thermal[1], _thermal[0], _sen[1], _sen[0]] for (_thermal, _sen) in
            zip(thermal_point_list, sen_point_list)]
    H_mat, flag_result = findHomoraphy(good)
    return H_mat, (thermal_point_list, sen_point_list, outliers_ref, outliers_sen)