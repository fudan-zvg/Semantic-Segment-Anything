<p align="center">
  <img src="./figures/SSA_title_00.png" alt="SSA Icon"/>
</p>

### [Official repo](https://github.com/fudan-zvg/Semantic-Segment-Anything)
> **[Semantic Segment Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything)**  
> Jiaqi Chen, Zeyu Yang, and Li Zhang  
> Zhang Vision Group, Fudan Univerisity

Our _**S**emantic **S**egment **A**nything (SSA)_ project enhances the [Segment Anything dataset(SA-1B)](https://segment-anything.com/) with a dense category annotation engine.
The SSA annotation engine is completely automated, requiring no human annotators. It has the ability to annotate using an open vocabulary and can classify the basic categories of the COCO and ADE20K datasets. 
This tool fills the gap in SA-1B's limited fine-grained semantic labeling, while also significantly reducing the need for manual annotation and associated costs. It has the potential to serve as a foundation for training large-scale visual perception models and more fine-grained CLIP models.
### 🤔 Why do we need SSA?
- Although SA-1B is a large image segmentation dataset with fine mask segmentation annotations, it lacks semantic annotations for training.
- While advanced close-set segmenters such as Oneformer, open-set segmenters like CLIPSeg, and image caption methods like BLIP can provide rich semantic annotations, they often lack the ability to capture fine edges effectively liken SAM.
- By combining the strengths of SA-1B's fine image segmentation annotations with the rich semantic annotations provided by these advanced models, we can provide more image segmentation dataset for dense category annotation.
### 🚄 Semantic segment anything engine
The SSA engine consists of three stages:
- **Step II: Close-set annotation.** We use a close-set semantic segmentation model trained on COCO and ADE20K datasets to segment the image and obtain rough category information. The predicted labels only include categories from COCO and ADE20K.

- **Step II: Open-vocabulary annotation.** We utilize an image captioning model to describe each cropped local region of the image corresponding to each mask, obtaining open-vocabulary categories.

- **Step III: Final decision.** We merge the close-set categories and open-vocabulary categories obtained in the previous two steps into a candidate category list. This list is then inputted, along with the cropped local image corresponding to each mask, into CLIPSeg. The model outputs the most likely category for each region.
### 📖 News
🔥 2023/04/09: The example of Semantic Segment Anything for SA-1B is released.  
🔥 2023/04/05: SA-1B is released.  

## Examples
![](./figures/sa_225091_pred_top1.png)
![](./figures/sa_225634_pred_top1.png)
![](./figures/sa_229896_pred_top1.png)
![](./figures/example.png)

## 💻 Requirements
- Python 3.7+
- CUDA 11.1+

## 🛠️ Installation
```bash
conda env create -f environment.yaml
conda activate ssa
python -m spacy download en_core_web_sm
```
## 🚀 Quick Start
### 1. Download SA-1B
Download the [SA-1B](https://segment-anything.com/) dataset and unzip it to the `data/sa_1b` folder.  

**Folder sturcture:**
```none
├── Semantic-Segment-Anything
├── data
│   ├── sa_1b
│   │   ├── sa_223775.jpg
│   │   ├── sa_223775.json
│   │   ├── ...
```
Run our semantic annotation script with 8 GPUs:
```bash
python scripts/stable_two_stage_multi_segmenter_clip_seg.py --data_dir=data/examples --out_dir=output --world_size=8 --save_img
```
For each mask, we add a new field (e.g. 'class_name': 'face') as follows:
```bash
{
    'bbox': [81, 21, 434, 666],
    'area': 128047,
    'segmentation': {
        'size': [1500, 2250],
        'counts': 'kYg38l[18oeN8mY14aeN5\\Z1>'
    }, 
    'predicted_iou': 0.9704002737998962,
    'point_coords': [[474.71875, 597.3125]],
    'crop_box': [0, 0, 1381, 1006],
    'id': 1229599471,
    'stability_score': 0.9598413705825806,
    'class_name': 'face',
    'class_proposals': ['face', 'person', 'sun glasses']
}
```
## 👍 Acknowledgement
- [Segment Anything](https://segment-anything.com/) provides the SA-1B dataset
- [HuggingFace](https://huggingface.co/) provides code and pre-trained models.
- [CLIPSeg](https://arxiv.org/abs/2112.10003), [OneFormer](https://arxiv.org/abs/2211.06220), [BLIP](https://arxiv.org/abs/2201.12086) and [CLIP](https://arxiv.org/abs/2103.00020) provide powerful semantic segmentation, classification, and image caption models

## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{semantic2023,
    title = {Semantic Segment Anything},
    author = {Jiaqi Chen and Zeyu Yang and Li Zhang},
    howpublished = {\url{https://github.com/fudan-zvg/Semantic-Segment-Anything}},
    year = {2023}
}
```
