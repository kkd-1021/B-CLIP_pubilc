



import os
import cv2

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('i_path', metavar='N', type=str, nargs='+',
                    help='input video fold path')
parser.add_argument('o_path', metavar='N', type=str, nargs='+',
                    help='processed video fold path')
args = parser.parse_args()
video_fold_path =args.i_path[0]
output_fold_path =args.o_path[0]


from ultralytics import YOLO
model_pose = YOLO('yolov8n-pose.pt')

def check_two_person(frame, frame_idx):
    yolo_results = model_pose.track(frame, persist=False)
    annotated_frame = yolo_results[0].plot()

    if len(yolo_results[0].boxes)<2:
        return False
    else:
        return True

#video_fold_path = "/mnt/hdd2/dengruikang/asd_datasets_new/update"
#output_fold_path = "/mnt/hdd2/dengruikang/new_videos"
file_list = os.listdir(video_fold_path)

print(file_list)
for video in file_list:

    index = file_list.index(video)

    video_name = os.path.basename(video)

    input = os.path.join(video_fold_path, video_name)
    cap = cv2.VideoCapture(input)
    print(video_name)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fps=3
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频帧宽度
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频帧高度
    output_path =os.path.join(output_fold_path, video_name)# 新视频文件路径
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    valid_frame_count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(frame_count)
        print(video_name)
        print(str(index)+'/'+str(len(file_list)))

        is_two_person=check_two_person(frame,frame_count)
        if is_two_person==True:
            valid_frame_count+=1
            out.write(frame)


    if valid_frame_count<180:
        out.release()
        os.remove(output_path)
        print(output_path)
        print("drop")
    else:
        out.release()
    cap.release()





