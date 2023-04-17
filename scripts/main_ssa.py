import os
import argparse
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from pipeline import semantic_segment_anything_inference, eval_pipeline, img_load
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL

import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--ckpt_path', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes'], help='specify the set of class names')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    args = parser.parse_args()
    return args
    
def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt_path).to(rank)
    mask_branch_model = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')
    # yoo can add your own semantic branch here, and modify the following code
    if args.dataset == 'ade20k':
        semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
        semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to(rank)
    elif args.dataset == 'cityscapes':
        semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_swin_large")
        semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_cityscapes_swin_large").to(rank)
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    if args.dataset == 'ade20k':
        filenames = [fn_.replace('.jpg', '') for fn_ in os.listdir(args.data_dir) if '.jpg' in fn_]
    elif args.dataset == 'cityscapes':
        sub_folders = [fn_ for fn_ in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, fn_))]
        filenames = []
        for sub_folder in sub_folders:
            filenames += [os.path.join(sub_folder, fn_.replace('.png', '')) for fn_ in os.listdir(args.data_dir + sub_folder) if '.png' in fn_]
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')
    for file_name in local_filenames:
        img = img_load(args.data_dir, file_name, args.dataset)
        if args.dataset == 'ade20k':
            id2label = CONFIG_ADE20K_ID2LABEL
        elif args.dataset == 'cityscapes':
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        else:
            raise NotImplementedError()
        with torch.no_grad():
            semantic_segment_anything_inference(file_name, args.out_dir, rank, img=img, save_img=args.save_img,
                                   semantic_branch_processor=semantic_branch_processor,
                                   semantic_branch_model=semantic_branch_model,
                                   mask_branch_model=mask_branch_model,
                                   dataset=args.dataset,
                                   id2label=id2label)
    if args.eval and rank==0:
        assert args.gt_path is not None
        eval_pipeline(args.gt_path, args.out_dir, args.dataset)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main,args=(args,),nprocs=args.world_size,join=True)
    else:
        main(0, args)