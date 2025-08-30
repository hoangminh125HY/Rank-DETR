from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

def register_my_dataset():
    """
    Đăng ký dataset custom theo format COCO (train.json, val.json)
    """
    # Thay đường dẫn theo chỗ bạn để dataset
    train_json = "/kaggle/working/annotations/train.json"
    val_json = "/kaggle/working/annotations/val.json"
    img_dir = "/kaggle/input/data-private-bus-car-truck/Private_DTS/Images/Images"

    # Register train
    register_coco_instances("my_dataset_train", {}, train_json, img_dir)
    # Register val
    register_coco_instances("my_dataset_val", {}, val_json, img_dir)

    # (Optional) metadata
    MetadataCatalog.get("my_dataset_train").thing_classes = ["Bus","Car","Person"]
    MetadataCatalog.get("my_dataset_val").thing_classes = ["Bus","Car","Person"]

    print("✅ Registered my_dataset_train and my_dataset_val successfully!")