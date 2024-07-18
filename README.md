# VCP-CLIP (Accepted by ECCV 2024)
![](figures/Framework.png)

**VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation**

Zhen Qu, Xian Tao, Mukesh Prasad, Fei Shen, Zhengtao Zhang, Xinyi Gong, Guiguang Ding

[Paper link](https://arxiv.org/pdf/2407.12276)

## Table of Contents
* [📖 Introduction](#introduction)
* [🔧 Environments](#environments)
* [📊 Data Preparation](#data-preparation)
* [🚀 Run Experiments](#run-experiments)
* [🔗 Citation](#citation)
* [🙏 Acknowledgements](#acknowledgements)
* [📜 License](#license)

## Introduction
This repository contains source code for VCP-CLIP implemented with PyTorch. 


Recently, large-scale vision-language models such as CLIP
have demonstrated immense potential in zero-shot anomaly segmentation (ZSAS) task, utilizing a unified model to directly detect anomalies
on any unseen product with painstakingly crafted text prompts. However, existing methods often assume that the product category to be
inspected is known, thus setting product-specific text prompts, which is
difficult to achieve in the data privacy scenarios. Moreover, even the same
type of product exhibits significant differences due to specific components
and variations in the production process, posing significant challenges to
the design of text prompts. In this end, we propose a visual context
prompting model (VCP-CLIP) for ZSAS task based on CLIP. The insight behind VCP-CLIP is to employ visual context prompting to activate CLIP’s anomalous semantic perception ability. In specific, we first
design a Pre-VCP module to embed global visual information into the
text prompt, thus eliminating the necessity for product-specific prompts.
Then, we propose a novel Post-VCP module, that adjusts the text embeddings utilizing the fine-grained features of the images. In extensive
experiments conducted on 10 real-world industrial anomaly segmentation
datasets, VCP-CLIP achieved state-of-the-art performance in ZSAS task.


## Environments
Create a new conda environment and install required packages.
```
conda create -n VCP_env python=3.9
conda activate VCP_env
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Experiments are conducted on a NVIDIA RTX 3090.


## Data Preparation




## Run Experiments


## Citation
Please cite the following paper if the code help your project:

```bibtex
@article{qu2024vcpclipvisualcontextprompting,
      title={VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation}, 
      author={Zhen Qu and Xian Tao and Mukesh Prasad and Fei Shen and Zhengtao Zhang and Xinyi Gong and Guiguang Ding},
      year={2024},
      eprint={2407.12276},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.12276}
}
```

## Acknowledgements


## License
The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).
