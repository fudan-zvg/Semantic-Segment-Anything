# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import sys
import json
from typing import Optional
import cv2
import torch
from PIL import Image
import mmcv
from mmdet.core.visualization.image import imshow_det_bboxes
import numpy as np
import pycocotools.mask as maskUtils

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from transformers import BlipProcessor, BlipForConditionalGeneration
from cog import BasePredictor, Input, Path, BaseModel

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

sys.path.insert(0, "scripts")
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from clip import clip_classification
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation
from blip import open_vocabulary_classification_blip


MODEL_CACHE = "model_cache"


class ModelOutput(BaseModel):
    json_out: Optional[Path]
    img_out: Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "default"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda")
        self.generator = SamAutomaticMaskGenerator(self.sam, output_mode="coco_rle")

        # semantic segmentation
        rank = 0
        # the following models are pre-downloaded and cached to MODEL_CACHE to speed up inference 
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(rank)

        self.oneformer_ade20k_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.oneformer_ade20k_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(rank)

        self.oneformer_coco_processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_coco_swin_large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.oneformer_coco_model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_coco_swin_large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(rank)

        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(rank)

        self.clipseg_processor = AutoProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to(rank)
        self.clipseg_processor.image_processor.do_resize = False

    def predict(
        self,
        image: Path = Input(description="Input image"),
        output_json: bool = Input(default=True, description="return raw json output"),
    ) -> ModelOutput:
        """Run a single prediction on the model"""

        img = cv2.imread(str(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = self.generator.generate(img)

        seg_json = "/tmp/mask.json"
        with open(seg_json, "w") as f:
            json.dump(masks, f)

        json_out = "/tmp/seg_out.json"
        seg_out = "/tmp/seg_out.png"

        semantic_annotation_pipeline(
            seg_json,
            str(image),
            json_out,
            seg_out,
            clip_processor=self.clip_processor,
            clip_model=self.clip_model,
            oneformer_ade20k_processor=self.oneformer_ade20k_processor,
            oneformer_ade20k_model=self.oneformer_ade20k_model,
            oneformer_coco_processor=self.oneformer_coco_processor,
            oneformer_coco_model=self.oneformer_coco_model,
            blip_processor=self.blip_processor,
            blip_model=self.blip_model,
            clipseg_processor=self.clipseg_processor,
            clipseg_model=self.clipseg_model,
        )

        return ModelOutput(
            json_out=Path(json_out) if output_json else None, img_out=Path(seg_out)
        )


def semantic_annotation_pipeline(
    seg_json,
    image,
    json_out,
    seg_out,
    rank=0,
    scale_small=1.2,
    scale_large=1.6,
    clip_processor=None,
    clip_model=None,
    oneformer_ade20k_processor=None,
    oneformer_ade20k_model=None,
    oneformer_coco_processor=None,
    oneformer_coco_model=None,
    blip_processor=None,
    blip_model=None,
    clipseg_processor=None,
    clipseg_model=None,
):
    anns = mmcv.load(seg_json)
    img = mmcv.imread(image)
    bitmasks, class_names = [], []
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(
        Image.fromarray(img), oneformer_coco_processor, oneformer_coco_model, 0
    )
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(
        Image.fromarray(img), oneformer_ade20k_processor, oneformer_ade20k_model, 0
    )

    for ann in anns:
        valid_mask = torch.tensor(maskUtils.decode(ann["segmentation"])).bool()
        # get the class ids of the valid pixels
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        top_k_coco_propose_classes_ids = (
            torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        )
        top_k_ade20k_propose_classes_ids = (
            torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        )
        local_class_names = set()
        local_class_names = set.union(
            local_class_names,
            set(
                [
                    CONFIG_ADE20K_ID2LABEL["id2label"][str(class_id.item())]
                    for class_id in top_k_ade20k_propose_classes_ids
                ]
            ),
        )
        local_class_names = set.union(
            local_class_names,
            set(
                (
                    [
                        CONFIG_COCO_ID2LABEL["refined_id2label"][str(class_id.item())]
                        for class_id in top_k_coco_propose_classes_ids
                    ]
                )
            ),
        )
        patch_small = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_small,
        )
        patch_large = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        patch_huge = mmcv.imcrop(
            img,
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        valid_mask_huge_crop = mmcv.imcrop(
            valid_mask.numpy(),
            np.array(
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
            ),
            scale=scale_large,
        )
        op_class_list = open_vocabulary_classification_blip(
            patch_large, blip_processor, blip_model, rank
        )
        local_class_list = list(
            set.union(local_class_names, set(op_class_list))
        )  # , set(refined_imagenet_class_names)
        mask_categories = clip_classification(
            patch_small,
            local_class_list,
            3 if len(local_class_list) > 3 else len(local_class_list),
            clip_processor,
            clip_model,
            rank,
        )
        class_ids_patch_huge = clipseg_segmentation(
            patch_huge, mask_categories, clipseg_processor, clipseg_model, rank
        ).argmax(0)
        top_1_patch_huge = (
            torch.bincount(
                class_ids_patch_huge[torch.tensor(valid_mask_huge_crop)].flatten()
            )
            .topk(1)
            .indices
        )
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        ann["class_name"] = str(top_1_mask_category)
        ann["class_proposals"] = mask_categories
        class_names.append(ann["class_name"])
        bitmasks.append(maskUtils.decode(ann["segmentation"]))

    mmcv.dump(anns, json_out)
    imshow_det_bboxes(
        img,
        bboxes=None,
        labels=np.arange(len(bitmasks)),
        segms=np.stack(bitmasks),
        class_names=class_names,
        font_size=25,
        show=False,
        out_file=seg_out,
    )
