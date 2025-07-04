# Guided Active Learning for Medical Image Segmentation

This repository contains code of the experiments conduced for the paper titled 'Guided Active Learning for Medical Image Segmentation' of the MICCAI 2025 conference.

A new version will be available soon.

## Installation
For dependencies please check environment.yml.
The installation has been tested on Ubuntu 18.04.5 LTS.

### Setting up using conda (some packages might not be optional)
```
conda create -n env_alunet python=3.9
conda activate env_alunet
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytables -c conda-forge
pip install opencv-python
pip install pynrrd
conda install -c anaconda h5py
pip install imblearn
pip install imutils
pip install keyboard
pip install xgboost
conda install -c conda-forge pingouin
pip install torchcontrib
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
cd submodlib
pip install .
pip install scikit-learn
git clone https://github.com/decile-team/submodlib.git
pip install kneed
conda install -c conda-forge umap-learn
conda remove krb5 --force -y
conda install krb5==1.15.1 --no-deps -y
```

Install nnUnet following the installation instructions [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

Ensure that the nnUnet path  [Setting up Paths](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md) are set to:
* nnUNet_raw=code_root/nnunet/data/nnunet/nnUNet_raw
* nnUNet_preprocessed=code_root/nnunet/data/nnunet/nnUNet_preprocessed
* nnUNet_results=code_root/nnunet/data/nnunet/nnUNet_results


## Dataset preparation
Please download imaging data for the Beyond the Cranial Vault (BTCV) Abdomen data set
[Multi-organ Abdominal CT Reference Standard Segmentations](https://zenodo.org/records/1169361)


## Run experiments
### Run targeted active learning experiment by selecting a dataset, sampling strategy, fold, budget and a function to execute
```
python TAL.py --dataset=CTA18 --dataset_name_or_id=900 --strategy=USIMFT --func=al --label_manual=True --targeted=True --segauto=False  --versionUse=-1
```

## Evaluation
After conduction active learning experiments, please check the output folder for trained nnUNet models and validation results.

## Contributing
Anonymized
