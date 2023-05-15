# EdgeSRGAN

[Generative Adversarial Super-Resolution at the Edge with Knowledge Distillation](https://arxiv.org/abs/2209.03355)

<img src="figures/results.png" alt= “Results” width="575">

## Abstract
Single-Image Super-Resolution can support robotic tasks in environments where a reliable visual stream is required to monitor the mission, handle teleoperation or study relevant visual details. In this work, we propose an efficient Generative Adversarial Network model for real-time Super-Resolution, called EdgeSRGAN. We adopt a tailored architecture of the original SRGAN and model quantization to boost the execution on CPU and Edge TPU devices, achieving up to 200 fps inference. We further optimize our model by distilling its knowledge to a smaller version of the network and obtain remarkable improvements compared to the standard training approach. Our experiments show that our fast and lightweight model preserves considerably satisfying image quality compared to heavier state-of-the-art models. Finally, we conduct experiments on image transmission with bandwidth degradation to highlight the advantages of the proposed system for mobile robotic applications.

<img src="figures/kd.png" alt= “KD” width="1000">

## Description
This repository allows to train and test EdgeSRGAN on different Single Image Super-Resolution datasets using adversarial training combined with feature-wise Knowledge Distillation.

## Installation
We suggest to use a virtual environment (conda, venv, ...)
```
git clone git@github.com:PIC4SeR/EdgeSRGAN.git
pip install -r requirements.txt
```

## Usage
Training
```
git clone git@github.com:PIC4SeR/EdgeSRGAN.git
pip install -r requirements.txt
```

## Examples
<img src="figures/samples.png?raw=True" alt= “Examples” width="1000">

## Acknowledgments
This repository is intended for research scopes. If you use it for your research, please cite our paper using the following BibTeX:
```
@article{angarano2023generative,
title = {Generative Adversarial Super-Resolution at the edge with knowledge distillation},
journal = {Engineering Applications of Artificial Intelligence},
volume = {123},
pages = {106407},
year = {2023},
issn = {0952-1976},
author = {Simone Angarano and Francesco Salvetti and Mauro Martini and Marcello Chiaberge}}
```
We would like to thank the Interdepartmental Center for Service Robotics [PIC4SeR](https://pic4ser.polito.it), Politecnico di Torino.
