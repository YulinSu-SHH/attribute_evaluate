# 属性分类评测工具
用于属性分类任务的评测。

## 简介
* 单标签：topk, precision, recall, f1
* 多标签：accuracy

## 依赖库
* torch
* numpy
* sklearn
* prettytable
* pandas

## 文件结构
    |-- root_folder
        |-- .DS_Store
        |-- evaluate.py
        |-- gt
        |   |-- gt_gender_code.jsonl
        |   |-- gt_st_fishing.jsonl
        |   |-- gt_st_pedestrian_angle.jsonl
        |-- pred
        |   |-- prediction_gender_code.jsonl
        |   |-- prediction_st_fishing.jsonl
        |   |-- prediction_st_pedestrian_angle.jsonl
        |-- result
            |-- result.csv
            |-- result.txt

## 数据格式
### gt
使用jsonl保存
>{
    "image_id": "4.jpg",
    "attribute": "prediction_st_pedestrian_angle",
    "labels": ["st_front", "st_side", "st_back"],
    "gt": [0, 1, 0]
}

## pred
使用jsonl保存
>{
    "image_id": "0.jpg",
    "attribute": "prediction_st_pedestrian_angle",
    "labels": ["st_front", "st_side", "st_back"],
    "pred": [0.33, 0.1, 0.3]
}

## 使用方法
* gt:GT的存放路径，文件以“gt_属性名”命名。
* pred:模型输出（预测概率）的存放路径，文件以“pred_属性名”命名。
* 运行python文件：
    ```python
    python evaluate.py --file_root='./' --topk=[1,3] --multilabel=False 
    ```
