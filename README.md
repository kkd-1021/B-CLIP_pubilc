
# Unveiling Early Autism Through Machine Learning: Retrospective and Prospective Analysis of Caregiver-Child Interaction Videos in Natural Settings



Code of B-CLIP, a novel machine learning model to classify 3-minute videos into ASD or non-ASD categories

    
# Environment Setup
To set up the environment, you can easily run the following command:
```
conda create -n B-CLIP python=3.7
conda activate B-CLIP
pip install -r requirements.txt
```

and install Apex as follows
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Data Preparation

- **Step \#1:prepare video set** 
- you should prepare a set of videos, each video is at least 3 minutes in length.
all the videos are put in a fold (for example, ./org_video_path)


- **Step \#2:video preprocess**  
- remove the useless frames using YOLO. 
```
python video_preprocess.py org_video_path processed_video_path
```
Then videos used in the following training and validation are in ./processed_video_path. 


-  **Step \#3:clinical information** 
- The clinical information should be noted in the following format, and stored as ./clinical_data/totaldata.xlsx. Noted that the XLSX should be in this format: 
```
name,diagnosis,gender,age(month),mullen:gross motor,mullen:visual reception,mullen:fine motor,mullen:receptive language,mullen:expressive language,SA CSS,RRB CSS
```
see ./clinical_data/totaldata.xlsx as an example.
- A videos' name list should be stored as ./clinical_data/total_video_list.txt. Noted that videos' name should be in this format: 
```
'child's name'+'index(if more than one video for a child)'+'_'+'.mp4'
```
see ./clinical_data/total_video_list.txt as an example.

-  **Step \#3:generate train and val labels**  
after preparing clinical data, run the script to generate video_labels
```
python generate_label/label_generator256_asdvstdbap.py
```

The ./video_labels fold should be in the following formate:
```
video_labels
    train_feat_new256_0.txt
    train_feat_new256_1.txt
    train_feat_new256_2.txt
    train_feat_new256_3.txt
    train_feat_new256_4.txt
    val_feat_new256_no_id_0.txt
    val_feat_new256_no_id_1.txt
    val_feat_new256_no_id_2.txt
    val_feat_new256_no_id_3.txt
    val_feat_new256_no_id_4.txt
    val_feat_new256_with_3id_0.txt
    val_feat_new256_with_3id_1.txt
    val_feat_new256_with_3id_2.txt
    val_feat_new256_with_3id_3.txt
    val_feat_new256_with_3id_4.txt
```


# Train
There is an example in configs/asd_tdbap_preprocessed/config0.yaml. Change the root path in this yaml file to the fold where processed videos in, and run the script
```
bash train.sh
```
- **Note:**
- --nproc_per_node = the number of gpu in your server

# Test
There is an example in configs/asd_tdbap_preprocessed_val/config0.yaml. Change the root path in this yaml file to the fold where processed videos in, and run the script
```
bash val.sh
```
**Note:**
- --nproc_per_node = the number of gpu in your server
- --resume the path to the checkpoint

# Interpretability
- **Step \#1**  
- You should firstly write a explain_list.txt including the video you want to explain. Put the file under ./video_labels
- See ./video_labels/explain_list.txt as an example


- **Step \#2** 
- You should also prepare a config file. There is an example in configs/explain_config.yaml. Change the root path in this yaml file to the fold where processed videos in, and run the script
```
bash explain.sh
```
- **Note:**
- --nproc_per_node = the number of gpu in your server
- --resume the path to the checkpoint
- you can find the Interpretability result in ./explain


- **Step \#3** 
- select the video id——the result fold name(number), for example, 10.0
run the script to generate a heatmap of the video corresponding to the fold name(number)
```
python draw_heatmap.py 10
```
you can find the heatmap and original video frames in ./explain

# Acknowledgements
Parts of the codes are borrowed from [X-CLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP),[mmaction2](https://github.com/open-mmlab/mmaction2), [Swin](https://github.com/microsoft/Swin-Transformer) and [CLIP](https://github.com/openai/CLIP). Sincere thanks to their wonderful works.
