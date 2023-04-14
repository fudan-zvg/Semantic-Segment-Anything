import torch
import torch.nn.functional as F

def oneformer_coco_segmentation(image, oneformer_coco_processor, oneformer_coco_model, rank):
    inputs = oneformer_coco_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_coco_model(**inputs)
    predicted_semantic_map = oneformer_coco_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_ade20k_segmentation(image, oneformer_ade20k_processor, oneformer_ade20k_model, rank):
    inputs = oneformer_ade20k_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_ade20k_model(**inputs)
    predicted_semantic_map = oneformer_ade20k_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map

def oneformer_cityscapes_segmentation(image, oneformer_cityscapes_processor, oneformer_cityscapes_model, rank):
    inputs = oneformer_cityscapes_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_cityscapes_model(**inputs)
    predicted_semantic_map = oneformer_cityscapes_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]])[0]
    return predicted_semantic_map