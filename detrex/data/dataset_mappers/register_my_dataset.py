# coding=utf-8
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Định nghĩa classes của dataset custom
MY_CATEGORIES = [
    {"id": 0, "name": "Bus"},
    {"id": 1, "name": "Car"},
    {"id": 2, "name": "Person"}
]

# Định nghĩa các split của dataset trên Kaggle
_PREDEFINED_SPLITS = {
    "my_dataset_train": (
        "/kaggle/working/dataset/train.json",
        "/kaggle/working/dataset/train/images",
    ),
    "my_dataset_val": (
        "/kaggle/working/dataset/val.json",
        "/kaggle/working/dataset/val/images",
    ),
}

def _get_my_dataset_meta():
    thing_ids = [k["id"] for k in MY_CATEGORIES]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in MY_CATEGORIES]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }

def register_my_dataset(root=None):  # Không cần root vì dùng đường dẫn tuyệt đối
    for key, (json_file, image_root) in _PREDEFINED_SPLITS.items():  # Đúng thứ tự
        register_coco_instances(
            key,
            _get_my_dataset_meta(),
            json_file,
            image_root,
        )

