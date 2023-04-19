# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.utils import print_log

import os
import mmcv
import argparse
import numpy as np
from collections import OrderedDict
import pycocotools.mask as maskUtils
from prettytable import PrettyTable
from torchvision.utils import save_image, make_grid
from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--gt_path', help='the directory of gt annotations')
    parser.add_argument('--result_path', help='the directory of semantic predictions')
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the dataset')
    args = parser.parse_args()
    return args

args = parse_args()
logger = None
if args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
    class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
elif args.dataset == 'ade20k':
    class_names = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')
file_client = mmcv.FileClient(**{'backend': 'disk'})
pre_eval_results = []
gt_path = args.gt_path
res_path = args.result_path
if args.dataset == 'cityscapes':
    prefixs = ['frankfurt','lindau','munster']
elif args.dataset == 'foggy_driving':
    prefixs = ['public', 'pedestrian']
elif args.dataset == 'ade20k':
    prefixs = ['']
else:
    raise NotImplementedError
for split in tqdm(prefixs, desc="Split loop"):
    gt_path_split = os.path.join(gt_path, split)
    res_path_split = os.path.join(res_path, split)
    filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
    for i, fn_ in enumerate(tqdm(filenames, desc="File loop")):
        pred_fn = os.path.join(res_path_split, fn_)
        result = mmcv.load(pred_fn)
        num_classes = len(class_names)
        init_flag = True
        for id_str, mask in result['semantic_mask'].items():
            mask_ = maskUtils.decode(mask)
            h, w = mask_.shape
            if init_flag:
                seg_mask = torch.zeros((1, 1, h, w))
                init_flag = False
            mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
            seg_mask[mask_] = int(id_str)
        seg_logit = torch.zeros((1, num_classes, h, w))
        seg_logit.scatter_(1, seg_mask.long(), 1)
        seg_logit = seg_logit.float()
        seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
        if args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json','_gtFine_labelTrainIds.png'))
        elif args.dataset == 'ade20k':
            gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json','.png'))
        img_bytes = file_client.get(gt_fn_)
        seg_map = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend='pillow').squeeze().astype(np.uint8)
        if args.dataset=='ade20k':
            seg_map = seg_map - 1
        pre_eval_results.append(intersect_and_union(
                        seg_pred,
                        seg_map,
                        num_classes,
                        255,
                        label_map=dict(),
                        reduce_zero_label=False))

ret_metrics = pre_eval_to_metrics(pre_eval_results, ['mIoU'])
ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
# each class table
ret_metrics.pop('aAcc', None)
ret_metrics_class = OrderedDict({
    ret_metric: np.round(ret_metric_value * 100, 2)
    for ret_metric, ret_metric_value in ret_metrics.items()
})
ret_metrics_class.update({'Class': class_names})
ret_metrics_class.move_to_end('Class', last=False)

# for logger
class_table_data = PrettyTable()
for key, val in ret_metrics_class.items():
    class_table_data.add_column(key, val)

summary_table_data = PrettyTable()
for key, val in ret_metrics_summary.items():
    if key == 'aAcc':
        summary_table_data.add_column(key, [val])
    else:
        summary_table_data.add_column('m' + key, [val])

print_log('per class results:', logger)
print_log('\n' + class_table_data.get_string(), logger=logger)
print_log('Summary:', logger)
print_log('\n' + summary_table_data.get_string(), logger=logger)
