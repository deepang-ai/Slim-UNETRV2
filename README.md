# Slim UNETRV2: 3D Image Segmentation for Resource-Limited Medical Portable Devices

> More details of this project will be released soon.

# Network Architecture

![Overview](./figures/Overview.jpg)

# Data Description
Dataset Name: BraTS2021

Modality: MRI

Size: 1470 3D volumes (1251 Training + 219 Validation)

Challenge: RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge

- Register and download the official BraTS 21 dataset from the link below and place then into "TrainingData" in the dataset folder:

  https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

  For example, the address of a single file is as follows:

  "TrainingData/BraTS2021_01146/BraTS2021_01146_flair.nii.gz"

- Download the json file from this [link](https://drive.google.com/file/d/1i-BXYe-wZ8R9Vp3GXoajGyqaJ65Jybg1/view?usp=sharing) and placed in the same folder as the dataset.

The sub-regions considered for evaluation in BraTS 21 challenge are the "enhancing tumor" (ET), the "tumor core" (TC), and the "whole tumor" (WT). The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (NCR) parts of the tumor. The appearance of NCR is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edematous/invaded tissue (ED), which is typically depicted by hyper-intense signal in FLAIR [[BraTS 21]](http://braintumorsegmentation.org/).


# Benchmark
## BraTS2021 dataset
Performance comparative analysis of different network architectures for brain tumor segmentation in the BraTS2021 dataset.
![Benchmark](./figures/Benchmark.png)

## MM-WHS dataset
Performance comparison for heart segmentation using the MM-WHS dataset.
![Benchmark2](./figures/Benchmark2.png)

# Visualization

## BraTS2021 dataset
Qualitative visualizations of the Slim UNETRV2 and baseline approaches under BraTS2021 segmentation task.
![Visualization](./figures/Visualization.png)

## MM-WHS dataset
Qualitative visualizations of the Slim UNETRV2 and baseline approaches under MMWHS heart segmentation task.

![Visualization2](./figures/Visualization2.png)