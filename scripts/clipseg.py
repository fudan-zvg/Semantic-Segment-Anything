import torch
import torch.nn.functional as F

def clipseg_segmentation(image, class_list, clipseg_processor, clipseg_model, rank):
    inputs = clipseg_processor(
        text=class_list, images=[image] * len(class_list),
        padding=True, return_tensors="pt").to(rank)
    # resize inputs['pixel_values'] to the longesr side of inputs['pixel_values']
    h, w = inputs['pixel_values'].shape[-2:]
    fixed_scale = (512, 512)
    inputs['pixel_values'] = F.interpolate(
        inputs['pixel_values'],
        size=fixed_scale,
        mode='bilinear',
        align_corners=False)
    outputs = clipseg_model(**inputs)
    logits = F.interpolate(outputs.logits[None], size=(h, w), mode='bilinear', align_corners=False)[0]
    return logits