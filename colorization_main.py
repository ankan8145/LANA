import sys,argparse
import pytorch_lightning as pl
# import lightning.pytorch as pl
from torch.utils.data import DataLoader
from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import time
import torch
from share import *



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-m",
        "--multicolor",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    parser.add_argument(
        "-s",
        "--usesam",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="continue train from resume",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args,_ = get_parser()

    if args.train:  
        n_gpu = 1
        # init_model_path = r"D:\Image Colorization\L-CAD checkpoint\init_model.ckpt"
        # init_model_path = r"D:\Image Colorization\L-CAD checkpoint\coco_weight.ckpt"
        init_model_path = r"C:\Users\thisi\Documents\image colorization\L-CAD - Copy1\lightning_logs\version_10\checkpoints\epoch=5-step=22266.ckpt"

        batch_size = 4
        logger_freq = 1000
        learning_rate = 1e-5 * n_gpu
        sd_locked = False # 
        only_mid_control = False

        model = create_model('configs/cldm_v15_ehdecoder.yaml').cpu()
        model.load_state_dict(load_state_dict(init_model_path, location='cpu'), strict=False)
        model.learning_rate = learning_rate
        model.sd_locked = sd_locked
        model.only_mid_control = only_mid_control  
        model.usesam=True
        
        dataset = MyDataset(img_dir= r"D:\Image Colorization\modified_coco",caption_dir= r'D:\Image Colorization\modified_coco\lcode_meta_data',use_sam=True) 
        
        dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
        logger = ImageLogger(batch_frequency=logger_freq)
        # trainer = pl.Trainer(gpus=n_gpu, precision=32, callbacks=[logger])
        trainer = pl.Trainer(precision=32, callbacks=[logger])
        # Train!
        trainer.fit(model, dataloader)

    else: # test or val
    
        # resume_path='.models/xxxxx.ckpt'
        resume_path= r"D:\Image Colorization\L-CAD checkpoint\coco_weight.ckpt"

        batch_size = 1 

        model = create_model('configs/cldm_v15_ehdecoder.yaml').cpu()
        model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
        
        trainer = pl.Trainer(precision=32)
        if args.multicolor: # test demo
            if args.usesam: # -m -s
                model.usesam = True
                dataset = MyDataset(img_dir='example', caption_dir='sam_mask', split='test',use_sam=True) 
                dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
                trainer.test(model, datamodule=dataloader)
            else: # -m
                model.usesam = False
                dataset = MyDataset(img_dir='example', caption_dir='example', split='test') 
                dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
                trainer.test(model, dataloader)
        else: # val
            model.usesam = False
            dataset = MyDataset(img_dir=r"D:\Image Colorization\modified_coco", caption_dir=r'D:\Image Colorization\modified_coco\lcode_meta_data', split='val') # 
            dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
            trainer.test(model, dataloader)

