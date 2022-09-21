import os
import pickle
import pandas

def filter_list(path_list):
    name_set=set()
    filtered_list=[]
    for path in path_list:
        name=os.path.split(path[0])[-1]
        if name not in name_set:
            filtered_list.append(path)
            name_set.add(name)
    return filtered_list



def csv_to_list(csvFileName):
    import csv
    with open(csvFileName,'r') as f:
        csvreader=csv.reader(f)
        final_list=list(csvreader)
        final_list=[_ for _ in final_list]
    return final_list


def exist_or_mkdir(dirlist):
    dirlist=[dirlist] if not isinstance(dirlist,list) else dirlist
    if isinstance(dirlist,list):
        for dir in dirlist:
            if not os.path.isdir(dir):
                os.makedirs(dir)

def file_exist_ornot(filelist,flag='and'):
    pass

def pickle_save(outpath,params):
    with open(outpath, 'wb') as f:
        pickle.dump(params, f)

def picckle_load(outpath):
    with open(outpath) as f:
        params=pickle.load(f)
    return params


def list_to_csv(csvPath,listFile):
    import pandas as pd
    pdFile=pd.DataFrame(listFile)
    pdFile.to_csv(csvPath,index=0,header=0)#可以用sep指定分隔符，默认为','，index,header表示是否添加行列索引
