from detectron2.data.datasets import register_coco_instances

# Train dataset
register_coco_instances(
    "my_dataset_train",
    {},
    "/kaggle/working/annotations/train.json",   # sửa đúng đường dẫn JSON
    "/kaggle/input/data-private-bus-car-truck/Private_DTS/Images/Images"  # sửa đúng đường dẫn ảnh
)

# Val dataset
register_coco_instances(
    "my_dataset_val",
    {},
    "/kaggle/working/annotations/val.json",     # sửa đúng đường dẫn JSON
    "/kaggle/input/data-private-bus-car-truck/Private_DTS/Images/Images"  # sửa đúng đường dẫn ảnh
)