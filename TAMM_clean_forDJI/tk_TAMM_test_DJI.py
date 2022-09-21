from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
from image_process import img_read
import cv2
import numpy as np
from pathlib import Path

global thermal_path,thermal_rgb_path,visible_path,output_path
global thermal_rgb_img,visible_img,thermal_img,mosaic_img,corres_img1,corres_img2
global level_max,patch_size,search_radius,thermal_upsample,crop_size

level_max=4
patch_size=40
search_radius=60


#_Z is
# thermal_upsample=2.1
# visible_upsample=1/3
#crop_size=920
#_W is
thermal_upsample=1.25
visible_upsample=1/2.05
crop_size=640


thermal_rgb_img = None
visible_img = None

global show_scale
show_scale=0.4

def callback_thermal():
    filename=filedialog.askopenfilename()
    global thermal_path,thermal_img
    thermal_path = filename

    tiff_img = img_read(thermal_path,scale=1)
    tiff_norm = cv2.normalize(tiff_img,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    tiff_norm = cv2.resize(tiff_norm,None,fx=show_scale,fy=show_scale)
    pilImage = Image.fromarray(tiff_norm)
    img = pilImage
    thermal_img = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=thermal_img)
    imglabel.grid(row=1,column=0)



def callback_thermal_rgb():
    filename=filedialog.askopenfilename()
    global thermal_rgb_path,thermal_rgb_img
    thermal_rgb_path = filename

    pilImage = Image.open(thermal_rgb_path)
    width = int(pilImage.size[0]*show_scale)
    height = int(pilImage.size[1]*show_scale)
    img = pilImage.resize((width, height), Image.ANTIALIAS)
    thermal_rgb_img = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=thermal_rgb_img)
    imglabel.grid(row=1,column=1)



def callback_visible():
    filename = filedialog.askopenfilename()
    global visible_path,visible_img
    visible_path = filename

    pilImage = Image.open(visible_path)
    width = int(pilImage.size[0]*show_scale*0.5)
    height = int(pilImage.size[1]*show_scale*0.5)
    img = pilImage.resize((width, height), Image.ANTIALIAS)
    visible_img = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=visible_img)
    imglabel.grid(row=1,column=2)



def callback_output():
    # filename = filedialog.asksaveasfilename()
    filename = filedialog.askdirectory()
    global level_max, patch_size, search_radius, thermal_upsample, crop_size
    global output_path
    global thermal_path, thermal_rgb_path, visible_path
    output_path = filename

    tk_t.insert("end", f"输入图像: \n"
                       f"thermal_path: {thermal_path}\n"
                       f"thermal_rgb_path: {thermal_rgb_path}\n"
                       f"visible_path: {visible_path}\n"
                )
    tk_t.insert("end", f"配准的相关参数: \n"
                       f"level_max:{level_max}\n"
                       f"patch_size:{patch_size}\n"
                       f"search_radius:{search_radius}\n"
                       f"thermal_upsample:{thermal_upsample}\n"
                       f"crop_size:{crop_size}\n"
                )
    tk_t.insert("end", f"output_path: {output_path}\n")

def callback_run():
    global thermal_path,thermal_rgb_path,visible_path,output_path,mosaic_img,corres_img1,corres_img2
    global level_max, patch_size, search_radius, thermal_upsample,crop_size
    from main_TAMM import one_img_process
    kwargs={}
    kwargs['level_max'] = level_max
    kwargs['patch_size'] = patch_size
    kwargs['search_radius'] = search_radius
    kwargs['thermal_upsample'] = thermal_upsample
    kwargs['visible_upsample'] = visible_upsample
    kwargs['crop_size'] = crop_size


    one_img_process(therma_img_path = thermal_path,
                    therma_rgb_path = thermal_rgb_path,
                    visible_img_path = visible_path,
                    out_img = output_path,
                    kwargs = kwargs)
    from image_process import homo_parse
    homo=homo_parse(homo_path = (Path(output_path)/'homo'/Path(thermal_path).stem).with_suffix('.csv'))
    tk_t.insert("end", f"homo: {homo}\n")

    pilImage = Image.open((Path(output_path)/'mosaic'/Path(thermal_path).stem).with_suffix('.png'))
    img = pilImage.resize((300, 300), Image.ANTIALIAS)
    mosaic_img = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=mosaic_img)
    imglabel.grid(row=4,column=1)


    pilImage_1 = Image.open((Path(output_path)/'matchpoints_img'/Path(thermal_path).stem).with_suffix('.jpg'))
    img = pilImage_1.resize((300, 300), Image.ANTIALIAS)
    corres_img1 = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=corres_img1)
    imglabel.grid(row=4,column=2)

    pilImage_2 = Image.open((Path(output_path) / 'matchpoints_img' / Path(thermal_path).stem).with_suffix('.png'))
    img = pilImage_2.resize((300, 300), Image.ANTIALIAS)
    corres_img2 = ImageTk.PhotoImage(img)
    imglabel = Label(root, image=corres_img2)
    imglabel.grid(row=4,column=3)

    tk_t.insert("end", f"process finished\n")






root=Tk()
root.title('热红外可见光影像配准系统')
Button(root,text='输入热红外文件(tiff)',command=callback_thermal).grid(row=0,column=0)
Button(root,text='输入热红外文件(rgb)',command=callback_thermal_rgb).grid(row=0,column=1)
Button(root,text='输入可见光文件',command=callback_visible).grid(row=0,column=2)
Button(root,text='结果保存地址',command=callback_output).grid(row=3,column=0)
Button(root,text='运行',command=callback_run).grid(row=3,column=1)
tk_t = Text(root,height=20,width=40)
tk_t.grid(row=4,column=0)

root.mainloop()
