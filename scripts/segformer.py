import torch.nn.functional as F

def segformer_segmentation(image, processor, model, rank):
    h, w, _ = image.shape
    inputs = processor(images=image, return_tensors="pt").to(rank)
    outputs = model(**inputs)
    logits = outputs.logits
    logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
    predicted_semantic_map = logits.argmax(dim=1).squeeze(0)
    return predicted_semantic_map