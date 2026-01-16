import os
import json
import xml.etree.ElementTree as ET
from shutil import copyfile
from tqdm import tqdm

# 定义VOC到COCO的转换函数
def voc_to_coco_classification(voc_folder, coco_folder, split="val"):
    # 创建COCO格式的数据字典
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # VOC 2007类别
    voc_classes = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
        "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
        "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    
    # 添加COCO格式的类别信息
    for i, class_name in enumerate(voc_classes[1:], 1):  # Skip "background"
        coco_data["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": class_name
        })
    
    # 获取验证集的图像列表
    image_dir = os.path.join(voc_folder, "JPEGImages")
    annotation_dir = os.path.join(voc_folder, "Annotations")
    split_file = os.path.join(voc_folder, "ImageSets", "Main", f"{split}.txt")
    
    with open(split_file, 'r') as f:
        image_ids = f.read().splitlines()
    
    annotation_id = 1
    image_id = 1
    
    # 遍历图像ID并转换
    for image_id_str in tqdm(image_ids):
        image_file = os.path.join(image_dir, f"{image_id_str}.jpg")
        annotation_file = os.path.join(annotation_dir, f"{image_id_str}.xml")
        
        # 复制图片到新的目录并修改为448x448
        new_image_path = os.path.join(coco_folder, f"{split}2014", f"{image_id_str}.jpg")
        if not os.path.exists(os.path.dirname(new_image_path)):
            os.makedirs(os.path.dirname(new_image_path))
        copyfile(image_file, new_image_path)
        
        # 添加图像信息到COCO格式
        coco_data["images"].append({
            "id": image_id,
            "file_name": f"{image_id_str}.jpg",
            "height": 448,  # 图像大小固定为448
            "width": 448,   # 图像大小固定为448
        })
        
        # 解析XML标注文件
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        
        # 遍历每个物体并提取注释
        for obj in root.iter('object'):
            class_name = obj.find('name').text
            class_id = voc_classes.index(class_name)
            
            # 只处理分类，不涉及bbox
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "area": 448 * 448,  # 假设图像区域
                "iscrowd": 0,
            })
            annotation_id += 1
        
        image_id += 1
    
    # 保存转换后的数据为COCO格式JSON
    if not os.path.exists(os.path.join(coco_folder, "annotations")):
        os.makedirs(os.path.join(coco_folder, "annotations"))
    
    with open(os.path.join(coco_folder, "annotations", f"instances_{split}2014.json"), 'w') as json_file:
        json.dump(coco_data, json_file)

# 运行转换函数
voc_folder = "VOC2007"  # VOC2007数据集路径
coco_folder = "VOC2007_COCO"  # 目标COCO格式保存文件夹路径
voc_to_coco_classification(voc_folder, coco_folder, 'train')
voc_to_coco_classification(voc_folder, coco_folder, 'val')
