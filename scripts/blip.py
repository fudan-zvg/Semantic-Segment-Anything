import torch
import torch.nn.functional as F
from utils import get_noun_phrases

def open_vocabulary_classification_blip(raw_image, blip_processor, blip_model, rank):
    # unconditional image captioning
    captioning_inputs = blip_processor(raw_image, return_tensors="pt").to(rank)
    out = blip_model.generate(**captioning_inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    ov_class_list = get_noun_phrases(caption)
    return ov_class_list