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
        "/kaggle/working/annotations",  # Đường dẫn tới thư mục images train
        "/kaggle/working/annotations/train.json",  # Đường dẫn tới file JSON train
    ),
    "my_dataset_val": (
        "/kaggle/working/annotations",  # Đường dẫn tới thư mục images val
        "/kaggle/working/annotations/val.json",  # Đường dẫn tới file JSON val
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
    for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
        register_coco_instances(
            key,
            _get_my_dataset_meta(),
            json_file,  # Sử dụng đường dẫn tuyệt đối
            image_root,  # Sử dụng đường dẫn tuyệt đối
        )

# Đăng ký dataset
register_my_dataset()