# CroMAM: A Cross-Magnification Attention Feature Fusion Model for Predicting Genetic Status and Survival of Gliomas using Histological Images

## 0. Obtaining the datasets from TCGA / Use HistoQC to preprocess WSI and obtain the usable area mask
#### histoqc : https://github.com/choosehappy/HistoQC

## 1. Extract patches from the whole slide images
```
python utils/patch_extraction.py --cancer=LGG --num-cpus=6 --magnification=10 --patch-size=224 --stratify=idh --wsi_path --wsi_mask_path --output_path
```

## 2. Create meta information; Split train/val dataset (5-fold)
```
python utils/create_meta_info.py --cancer=LGG --ffpe-only --magnification=10 --stratify=idh --root= --patch_root=
```
#### meta_csv file has two columns:  submitter_id， xx_status(idh_status)
#### meta_csv file name : meta_clinical_xxx_xxx.csv(meta_clinical_LGG_idh.csv)

## 3. Train/val the deep learning model
```
python main.py --cancer=LGG --outcome=idh -m=train --model_name=CroMAM -b=8 -vb=1 --repeats-per-epoch=8 --num-patches=8 --num-val=64 --save-interval=50 --magnification=5,10 --lr-backbone=3e-7 --lr-fusion=3e-5 --lr-classifier=3e-5 --epochs=100 --num-workers=4 --dropout=0.2 --patience=10 --branches=2  --checkpoint-dir= --pretrain
```
