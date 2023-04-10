<p align="center">
  <img src="./figures/SSA_title.png" alt="SSA Icon"/>
</p>

### [Official repo](https://github.com/fudan-zvg/Semantic-Segment-Anything)
> **[Semantic Segment Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything)**  
> Jiaqi Chen, Zeyu Yang, and Li Zhang  
> Zhang Vision Group, Fudan Univerisity

_**S**emantic **S**egment **A**nything (SSA)_ project enhances the [Segment Anything dataset(SA-1B)](https://segment-anything.com/) with a dense category annotation engine.
SSA is an automated annotation engine that serves as the initial semantic labeling for the SA-1B dataset. While human review and refinement may be required for more accurate labeling.
Thanks to the combined architecture of close-set segmentation and open-vocabulary segmentation, SSA produces satisfactory labeling for most samples and has the capability to provide more detailed annotations using image caption method.
This tool fills the gap in SA-1B's limited fine-grained semantic labeling, while also significantly reducing the need for manual annotation and associated costs. 
It has the potential to serve as a foundation for training large-scale visual perception models and more fine-grained CLIP models.
![](./figures/SSA_motivation.png)
### 🤔 Why do we need SSA?
- SA-1B is the largest image segmentation dataset to date, providing fine mask segmentation annotations. However, it does not provide category annotations for each mask, which are essential for training a semantic segmentation model.
- Advanced close-set segmenters like Oneformer, open-set segmenters like CLIPSeg, and image caption methods like BLIP can provide rich semantic annotations. However, their mask segmentation predictions may not be as comprehensive and accurate as the mask annotations in SA-1B.
- Therefore, by combining the fine image segmentation annotations of SA-1B with the rich semantic annotations provided by these advanced models, we can provide a more densely categorized image segmentation dataset.
### 👍 What SSA can do?
- **SSA + SA-1B:** SSA provides open-vocabulary and dense mask-level category annotations for large-scale SA-1B dataset. After manual review and refinement, these annotations can be used to train segmentation models or fine-grained CLIP models.
- **SSA + SAM:** This combination can provide detailed segmentation masks and category labels for new data, while keeping manual labor costs relatively low. Users can first run SAM to obtain mask annotations, and then input the image and mask annotation files into SSA to obtain category labels.
### 🚄 Semantic segment anything engine
![](./figures/SSA_model.png)
The SSA engine consists of three components:
- **(I) Close-set semantic segmentor (green).** Two close-set semantic segmentation models trained on COCO and ADE20K datasets respectively are used to segment the image and obtain rough category information. The predicted categories only include simple and basic categories to ensure that each mask receives a relevant label.
- **(II) Open-vocabulary classifier (blue).** An image captioning model is utilized to describe the cropped image patch corresponding to each mask. Nouns or phrases are then extracted as candidate open-vocabulary categories. This process provides more diverse category labels.
- **(III) Final decision module (orange).** The SSA engine uses a Class proposal filter (_i.e._ a CLIP) to filter out the top-_k_ most reasonable predictions from the mixed class list. Finally, the Open-vocabulary Segmentor predicts the most suitable category within the mask region based on the top-_k_ classes and image patch.

### 📖 News
🔥 2023/04/10: Semantic Segment Anything is released.  
🔥 2023/04/05: SA-1B is released.  

## Examples
![](./figures/sa_225091_class_name.png)
![](./figures/sa_225172_class_name.png)
![](./figures/sa_230745_class_name.png)
![](./figures/sa_227097_class_name.png)
- Addition example for Open-vocabulary annotations

![](./figures/SSA_open_vocab.png)

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
### 1. Download SA-1B dataset
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
Run our Semantic annotation engine with 8 GPUs:
```bash
python scripts/main.py --data_dir=data/examples --out_dir=output --world_size=8 --save_img
```
For each mask, we add two new fields (e.g. 'class_name': 'face' and 'class_proposals': ['face', 'person', 'sun glasses']). The class name is the most likely category for the mask, and the class proposals are the top-_k_ most likely categories from Class proposal filter. _k_ is set to 3 by default.
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
## 📈 Future work
We hope that excellent researchers in the community can come up with new improvements and ideas to do more work based on SSA. Some of our ideas are as follows:
- (I) The masks in SA-1B are often in three levels: whole, part, and subpart, 
and SSA often cannot provide accurate descriptions for too small part or subpart regions. Instead, we use broad categories. For example, SSA may predict "person" for body parts like neck or hand. 
Therefore, an architecture for more detailed semantic prediction is needed.
- (II) SSA is an ensemble of multiple models, which makes the inference speed slower compared to end-to-end models. 
We look forward to more efficient designs in the future. 

## 😄 Acknowledgement
- [Segment Anything](https://segment-anything.com/) provides the SA-1B dataset.
- [HuggingFace](https://huggingface.co/) provides code and pre-trained models.
- [CLIPSeg](https://arxiv.org/abs/2112.10003), [OneFormer](https://arxiv.org/abs/2211.06220), [BLIP](https://arxiv.org/abs/2201.12086) and [CLIP](https://arxiv.org/abs/2103.00020) provide powerful semantic segmentation, image caption and classification models.

## 📜 Citation
If you find this work useful for your research, please cite our github repo:
```bibtex
@misc{chen2023semantic,
    title = {Semantic Segment Anything},
    author = {Chen, Jiaqi and Yang, Zeyu and Zhang, Li},
    howpublished = {\url{https://github.com/fudan-zvg/Semantic-Segment-Anything}},
    year = {2023}
}
```
