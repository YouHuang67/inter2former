# Inter2Former
## Installation

To install the required dependencies, follow the detailed instructions in the [Installation](./INSTALL.md).

## Datasets

To prepare the datasets, follow the detailed instructions in the [Datasets](./DATASETS.md).

## Pre-trained Models

We provide pre-trained models so that you can start testing immediately:

[Pre-trained Models](https://mega.nz/file/Z7lxRAya#B0xOGs97dI4Qi_QRI2-3qLghmvKUQKm-4RF_UIbg54U)

The zip file contains the following model weight:
- `inter2former_sa1b_hq.pth`: SAM-distilled/HQ-trained Inter2Former weights (unzip to `work_dirs/inter2former_eval`)


## Training the Model

Training code will be released in future updates.

## Model Evaluation

To evaluate the pre-trained model, use one of the following commands:

```bash
bash tools/dist_test_no_viz.sh configs/datasets/eval_davis.py work_dirs/inter2former_eval/inter2former_sa1b_hq.pth 4 -c configs/eval_custom/ts1024.py
```
(Input resolution: 1024×1024)

or

```bash
bash tools/dist_test_no_viz.sh configs/datasets/eval_davis.py work_dirs/inter2former_eval/inter2former_sa1b_hq.pth 4 -c configs/eval_custom/ts2048.py
```
(Input resolution: 2048×2048)
