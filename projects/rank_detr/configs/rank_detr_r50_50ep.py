from detrex.config import get_config
from .models.rank_detr_r50 import model
import sys
sys.path.append("/kaggle/working/Rank-DETR/detrex/data/dataset_mappers")

from register_my_dataset import register_my_dataset

# Khởi tạo dataloader
dataloader = get_config("common/data/coco_detr.py").dataloader
dataloader.train.dataset.names = "my_dataset_train"
dataloader.test.dataset.names = "my_dataset_val"

# Các config khác
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train
# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/rank_detr_r50_50ep"

# max training iterations
train.max_iter = 12500

# run evaluation every 5000 iters
train.eval_period = 2500

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = 2500

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 4

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
