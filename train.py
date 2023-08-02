import torch
from torchvision.models.detection import MaskRCNN
from inference import Config
from dataloader import ImageHuBMAPDataset
from torch.utils.data import DataLoader
from utils import get_file_dir, convnext_fpn_backbone, Trainer
from utilities.augment import train_aug, valid_aug



train_dataset = ImageHuBMAPDataset(Config.root_dir + 'train/',Config.root_dir +'polygons.jsonl',tfms = train_aug,split ='train', ignore_label = 255, config = Config)
train_loader = DataLoader(train_dataset, batch_size=Config.train_bs, 
                           shuffle=True,   pin_memory=True, 
                           drop_last=True, collate_fn=lambda x: tuple(zip(*x)))

val_dataset = ImageHuBMAPDataset(Config.root_dir + 'train/',Config.root_dir +'polygons.jsonl',tfms = valid_aug,split ='val', ignore_label = 255, config = Config)

val_loader = DataLoader(val_dataset, batch_size=Config.train_bs, 
                           shuffle=False,  pin_memory=True, 
                           drop_last=True, collate_fn=lambda x: tuple(zip(*x)))


backbone = convnext_fpn_backbone(
    Config.backbone,
    Config.trainable_layers
)

model = MaskRCNN(
    backbone, 
    num_classes=Config.num_classes, 
    max_size=Config.max_size,
    min_size=Config.min_size,)

model.to(Config.device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    params, 
    lr=Config.lr, 
    weight_decay=Config.weight_decay
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=Config.step_size,
    gamma=Config.gamma
)
scaler = torch.cuda.amp.GradScaler()

trainer = Trainer(
    optimizer=optimizer,
    max_epochs=Config.epochs,
    device=Config.device,
    scaler=scaler,
    verbose_num=Config.verbose_num,
    split_size=Config.split_size,
    val_bs=Config.val_bs
)

history = trainer.fit(
    model, 
    train_dataloader = train_loader, 
    val_dataloader =  val_loader,
    ckpt_path = Config.save_path + Config.model_name + ".pth"
)

