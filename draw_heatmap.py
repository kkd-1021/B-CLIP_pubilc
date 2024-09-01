import numpy as np
import re
import os
import cv2

class PredPoint():
    def __init__(self, image_idx, x, y, ctb):
        self.image_idx = image_idx
        self.x = x
        self.y = y
        self.contribution = ctb

def get_bk_img_filenames(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        # Check if the filename matches the pattern "bk_img[number].jpg"
        if re.match(r"bk_\d+asd\.jpg", filename):
            filenames.append(os.path.join(folder_path,filename))
    return filenames
def read_explain_result_file(file_path):
    pred_point_list = []
    with open(file_path, 'r') as f_read:
        for line in f_read:
            if "pred" in line:
                continue
            image_idx = line.split(",")[0]
            x = int(float(line.split(",")[1]))
            y = int(float((line.split(",")[2]).split("/")[0]))
            contribution = float(line.split("/")[1])
            pred_point=PredPoint(image_idx, x, y, contribution)
            pred_point_list.append(pred_point)
    return pred_point_list
def get_normal_color(img):
    max = np.max(img)
    min = np.min(img)
    if max!=min:
        ret = (img - min) / (max - min) * 255
    else:
        ret = (img - min) / 0.000000001 * 255
    return ret

def get_normal_color_mid(img):

    max = np.max(img)
    min = np.min(img)

    pos=np.maximum(img, 0)
    neg = np.maximum(-img, 0)

    pos=pos/max*255/2
    neg =neg / (-min) * 255 / 2


    ret=255/2+pos-neg

    return ret
def keep_top_x_unique(matrix,top_x,top_num):
    topx=top_x
    unique_values, counts = np.unique(matrix.flatten(), return_counts=True)
    print("unique_values")
    print(matrix)
    print(unique_values)
    #top5_values = unique_values[np.argsort(counts)][::-1][:topx]
    sorted_list = sorted(unique_values, reverse=True)
    top_values = sorted_list[:topx]
    print("top value")
    print(top_values)

    matching_elements = np.isin(matrix, top_values)
    count = np.count_nonzero(matching_elements)
    print(count)
    if count<2:
        print("top x error")
        exit(1)

    while count>top_num*24*24 and topx>1:
        topx=topx-1
        top_values = sorted_list[:topx]
        matching_elements = np.isin(matrix, top_values)
        count = np.count_nonzero(matching_elements)
        print(count)
    print("top_value_count")
    print(count)
    mask = np.isin(matrix, top_values, invert=True)
    #print(mask)
    matrix[mask] = 0

    return matrix


def stitch_images(images, cols=6, rows=4):

  sub_img_h, sub_img_w, _ = images[0].shape
  stitch_img = np.zeros((rows * sub_img_h, cols * sub_img_w, 3), dtype=np.uint8)
  for i in range(rows):
    for j in range(cols):
      index = i * cols + j
      stitch_img[i * sub_img_h:(i + 1) * sub_img_h,
                 j * sub_img_w:(j + 1) * sub_img_w, :] = images[index]

  return stitch_img


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='person id to explain')
args = parser.parse_args()
person_id=args.integers[0]
step_len = 4

img_dir="./explain"
point_list_dir='./explain'

point_list_path=os.path.join(point_list_dir, "explain_result"+str(person_id)+".0.txt")
img_path_list=get_bk_img_filenames(os.path.join(img_dir, "bk_"+str(person_id)+".0"))

bk_img_list=[]
for img_path in img_path_list:
    img=cv2.imread(img_path)
    bk_img_list.append(img)


pred_block_list=read_explain_result_file(point_list_path)



sorted_pred_block_list = sorted(pred_block_list, key=lambda x: x.contribution)

#top_k_pred_block_list=sorted_pred_block_list
#for i in range(20,len(top_k_pred_block_list)):
#    top_k_pred_block_list[i].contribution=0



contr_map_list=[]
for i in range(24):
    contr_map=np.zeros((224,224),type).astype(np.float32)
    contr_map_list.append(contr_map)

for block in sorted_pred_block_list:
    contr_map_list[int(block.image_idx)][int(block.x-step_len/2):int(block.x+step_len/2),int(block.y-step_len/2):int(block.y+step_len/2)]=block.contribution
    print(block.contribution)


heatmap_list=[]
heatmap_topK_list=[]


contr_map_np=np.array(contr_map_list)
contr_map_np=get_normal_color_mid(-contr_map_np)


for i in range(24):
    #contr_map = get_normal_color(contr_map_np[i])

    contr_map = cv2.convertScaleAbs(contr_map_np[i])
    #color_map = cv2.applyColorMap(contr_map, cv2.COLORMAP_JET)
    #color_map= cv2.applyColorMap(contr_map, cv2.COLORMAP_CIVIDIS)  # 将cam的结果转成伪彩色图片
    color_map = cv2.applyColorMap(contr_map, cv2.COLORMAP_TWILIGHT_SHIFTED)
    heatmap = np.float32(color_map) / 255.  # 缩放到[0,1]之间
    bk_img = bk_img_list[i] / 255.
    # print(bk_img)

    cam = heatmap*0.8+ bk_img*0.2
    cam = cam / np.max(cam)
    cam = cam*255
    heatmap_list.append(cam)

heatmap=stitch_images(heatmap_list)
bkmap=stitch_images(bk_img_list)
for i in range(len(heatmap_list)):
    cv2.imwrite("./heatmap/heatmap_" + str(person_id) + "/"+str(i)+".jpg", heatmap_list[i])
cv2.imwrite("./heatmap/heatmap_"+str(person_id)+".jpg",heatmap)
cv2.imwrite("./heatmap/bk_"+str(person_id)+".jpg",bkmap)




