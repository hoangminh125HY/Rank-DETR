from detectron2.data.datasets import register_coco_instances

def register_my_dataset():
    """
    Register custom dataset (COCO format) into Detectron2/Detrex.
    """
    # Train set
    register_coco_instances(
        "my_dataset_train",   # name of dataset (used in config)
        {},
        "/kaggle/working/dataset/train/annotations.json",   # path to train json
        "/kaggle/working/dataset/train/images"              # path to train images
    )

    # Val set
    register_coco_instances(
        "my_dataset_val",
        {},
        "/kaggle/working/dataset/val/annotations.json",     # path to val json
        "/kaggle/working/dataset/val/images"
    )
