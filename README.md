# SYSU-HCP-at-ImageCLEF-VQA-Med-2021
<!-- This repository is the official implementation of paper [HCP-MIC at VQA-Med 2020: Effective Visual Representation for Medical Visual Quesion Answering](http://ceur-ws.org/Vol-2696/paper_74.pdf). -->

<!-- ## Citing this repository
If you find this code useful in your work, please consider citing us:

```
@inproceedings{chen2020hcp-mic,
  author    = {Guanqi Chen and
               Haifan Gong and
               Guanbin Li},
  title     = {{HCP-MIC} at VQA-Med 2020: Effective Visual Representation for Medical Visual Question Answering},
  booktitle = {Working Notes of {CLEF} 2020 - Conference and Labs of the Evaluation Forum, Thessaloniki, Greece, September 22-25, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2696},
  year      = {2020},
}
``` -->

## Main requirements

  * **torch == 1.4.0**
  * **torchvision == 0.5.0**
  * **tensorboardX == 2.1**
  * **Python 3**

The environment can be created following this:
```bash
# First, create a virtual environment
conda create -n your_env_name python=3.6
conda activate your_env_name

# Second, install the required packages from requirements.txt
conda install pip
pip install -r requirements.txt
```

## Models

The [ResNeSt](https://github.com/zhanghang1989/ResNeSt) needs to be installed before using.

## Datasets

The training set, validation set and test set can be downloaded from the [Google Drive](https://drive.google.com/file/d/1ovF5HF4t49ZJ8YMmaydfvcfeOf08h3yR/view?usp=sharing) directly and should be put in the `data` folder. 

## Training

You can train the model from scratch. The command and corresponding parameters are as follows:
```bash
python train.py [-input_size <int>] [-batch_size <int>] [-backbone <model_name>] [-train_set <Med_LT_train or train>] [-gpu <int>] [-world_size <int>] [-port <int>] [--mixup] [--label_smooth] [--superloss] [-dryrun] [-resume_path <checkpoint.pth>] [-pretrain <checkpoint.pth>]
```

## Evaluation

You can evaluate the trained model and save the results in a csv file.
```bash
python validation.py -model_path <checkpoint.pth> -csv_path <path_to_save_evaluation_results> [-input_size <int>] [-backbone <model>] [-gpu <int>]
```

## Pretrained models for VQA-Med 2021

We provide the pretrained models for VQA-Med 2021 in [Google Drive](https://drive.google.com/file/d/1tsTlAD6VDVhTd-9ocLEmgcoXZKBJqdnS/view?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1Q-P4mqmq2jFDQ2DyYenbqw) (code:i1nn). 

After downloading the models, you can get the result by the following command:
```bash
python inference_ensemble_once.py
```

<!-- The BBN is mainly modified from [BBN](https://github.com/Megvii-Nanjing/BBN), Bio-Bert pretrain is obtained from [Biobert](https://github.com/dmis-lab/biobert), the pickle data should be under the ```BBN-BioBert-Inference/data/``` folder.  -->

<!-- ## Contacts
If you have any questions about our work, please do not hesitate to contact us by emails.

Haifan Gong: haifangong@outlook.com -->