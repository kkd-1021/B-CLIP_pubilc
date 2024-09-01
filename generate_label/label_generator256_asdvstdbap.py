from copy import deepcopy

import pandas as pd
from label_utils import get_person_name,find_value_from_xlsx
import random
import os
child_map=[]
def get_id(child_name):
    for id in range(len(child_map)):
        if child_name==child_map[id]:
            return id
    child_map.append(child_name)

    return len(child_map)-1

class Person_storage():
    def __init__(self):
        self.person_list=[]


    def add_video(self, video_name):
        person_name = get_person_name(video_name.split("_")[0])
        #print("person_name: "+person_name)

        is_new_person=True
        for person in self.person_list:
            if person.person_name==person_name:
                person.add_video(video_name)
                is_new_person=False
                break

        if is_new_person is True:
            print("new person_name: " + person_name)
            new_person=Person(person_name)
            new_person.add_video(video_name)
            self.person_list.append(new_person)



class Person():
    def __init__(self,person_name):
        self.person_name=person_name
        self.video_list=[]
        self.label='TD'



    def add_video(self,video_name):
        self.video_list.append(video_name)

    def set_label(self,label):
        if label!='ASD' and label!='TD' and label!='BAP':
            print("label error")
            exit(1)
        self.label=label


def split_list(lst, n):
   
    avg = len(lst) / float(n)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out



f=open('./clinical_data/total_video_list.txt', 'r')
person_storage=Person_storage()
for txt_data in f:
    video_name = txt_data.strip('\n')
    person_storage.add_video(video_name)

print("person number= "+str(len(person_storage.person_list)))


io = r'./clinical_data/totaldata.xlsx'
excel_data = pd.read_excel(io, sheet_name = 0)#指定读取第一个sheet

asd_person_list=[]
bap_person_list=[]
td_person_list=[]


for person in person_storage.person_list:

    asd_ret = find_value_from_xlsx('diagnosis', excel_data, person.person_name)
    person.set_label(asd_ret)


    if asd_ret=='ASD':
        asd_person_list.append(person)
    if asd_ret=='BAP':
        bap_person_list.append(person)
    if asd_ret=='TD':
        td_person_list.append(person)

dist_path="./video_labels/"
copy_num=5

random.shuffle(asd_person_list)
random.shuffle(bap_person_list)
random.shuffle(td_person_list)
asd_list_set = split_list(asd_person_list, 5)
bap_list_set = split_list(bap_person_list, 5)
td_list_set = split_list(td_person_list, 5)


for split_idx in range(0,5):
    print(len(asd_list_set[split_idx]),len(bap_list_set[split_idx]),len(td_list_set[split_idx]))
    val_set=asd_list_set[split_idx]+bap_list_set[split_idx]+td_list_set[split_idx]
    print(len(val_set))
    # 将选择的元素从原列表中移除，剩余的元素为另一类
    train_set = [element for element in person_storage.person_list if element not in val_set]

    txt_file_name = dist_path + 'train_feat_new256_' + str(split_idx) + '.txt'
    if os.path.exists(txt_file_name):
        print("file already exists")
        exit(1)

    txt_file_name = dist_path + 'val_feat_new256_no_id_' + str(split_idx) + '.txt'
    if os.path.exists(txt_file_name):
        print("file already exists")
        exit(1)

    txt_file_name = dist_path + 'val_feat_new256_with_3id_' + str(split_idx) + '.txt'
    if os.path.exists(txt_file_name):
        print("file already exists")
        exit(1)










    mullen_feat_set=['mullen:gross motor','mullen:visual reception','mullen:fine motor','mullen:receptive language','mullen:expressive language']
    css_feat_set=['SA CSS','RRB CSS']


    for itr_person in train_set:
        person=deepcopy(itr_person)
        while len(person.video_list) < 6 :
            '''Each person should fill in at least 6 videos'''
            person.video_list.append(random.choice(person.video_list))

        total_feat=0

        asd_ret=find_value_from_xlsx('diagnosis',excel_data,person.person_name)
        if asd_ret=='ASD':
            total_feat=1
        else:
            if asd_ret=='BAP':
                total_feat=0
            else:
                if asd_ret=='TD':
                    total_feat=0
                else:
                    print("reding error")
                    exit(1)

        for mullen_feat in mullen_feat_set:
            mullen_ret=find_value_from_xlsx(mullen_feat,excel_data,person.person_name)

            if mullen_ret<=35:
                new_feat=1
            else:
                new_feat=0

            total_feat=total_feat*10+new_feat

        for css_feat in css_feat_set:
            css_feat_ret = find_value_from_xlsx(css_feat, excel_data, person.person_name)

            if css_feat_ret <=3:
                new_feat=0
            else:
                if css_feat_ret>3 and css_feat_ret<=5:
                    new_feat=1
                else:
                    if css_feat_ret>5:
                        new_feat=1
                    else:
                        print(person.person_name)
                        print(css_feat_ret)
                        print("css reding error")
                        exit(1)


            total_feat = total_feat * 10 + new_feat

        num_str = str(total_feat)
        num_length = len(num_str)

        '''reading check'''
        if num_length > 8:
            print("feat too long error")
            exit(1)

        if asd_ret == 'ASD':
            if total_feat // (10 ** 7) != 1:
                print("trainset feat error")
                exit(1)
        if asd_ret == 'TD':
            if total_feat // (10 ** 7) != 0:
                print("trainset feat error")
                exit(1)

        txt_file_name = dist_path + 'train_feat_new256_' + str(split_idx) + '.txt'
        with open(txt_file_name, 'a') as f:
            for video_name in person.video_list:
                f.write(video_name)
                f.write(' ')
                f.write(str(total_feat))
                f.write('\n')
        f.close()



    for person in val_set:

        total_feat=0
        asd_ret=find_value_from_xlsx('diagnosis',excel_data,person.person_name)
        if asd_ret=='ASD':
            total_feat=1
        else:
            if asd_ret=='BAP':
                total_feat=0
            else:
                if asd_ret=='TD':
                    total_feat=0
                else:
                    print("diagnosis reading error")
                    exit(1)

        for mullen_feat in mullen_feat_set:
            mullen_ret=find_value_from_xlsx(mullen_feat,excel_data,person.person_name)

            if mullen_ret<=35:
                new_feat=1
            else:
                new_feat=0

            total_feat=total_feat*10+new_feat

        for css_feat in css_feat_set:
            css_feat_ret = find_value_from_xlsx(css_feat, excel_data, person.person_name)

            if css_feat_ret <=3:
                new_feat=0
            else:
                if css_feat_ret>3 and css_feat_ret<=5:
                    new_feat=1
                else:
                    if css_feat_ret>5:
                        new_feat=1
                    else:
                        print(person.person_name)
                        print(css_feat_ret)
                        print("css reading error")
                        exit(1)


            total_feat = total_feat * 10 + new_feat

        num_str = str(total_feat)
        num_length = len(num_str)

        '''check'''
        if num_length > 8:
            print("feat too long error")
            exit(1)

        if asd_ret == 'ASD':
            if total_feat // (10 ** 7) != 1:
                print("trainset feat error")
                exit(1)
        if asd_ret == 'TD':
            if total_feat // (10 ** 7) != 0:
                print("trainset feat error")
                exit(1)

        txt_file_name = dist_path + 'val_feat_new256_no_id_' + str(split_idx) + '.txt'
        with open(txt_file_name, 'a') as f:
            for video_name in person.video_list:
                for i in range(copy_num):
                    f.write(video_name)
                    f.write(' ')
                    f.write(str(total_feat))
                    f.write('\n')
        f.close()

        feats_with_id = total_feat * 1000 + get_id(video_name)  # 最后3位为id

        txt_file_name = dist_path + 'val_feat_new256_with_3id_' + str(split_idx) + '.txt'
        with open(txt_file_name, 'a') as f:
            for video_name in person.video_list:
                for i in range(copy_num):
                    f.write(video_name)
                    f.write(' ')
                    f.write(str(feats_with_id))
                    f.write('\n')
        f.close()

