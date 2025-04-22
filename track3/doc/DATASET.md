# Track3 Dataset

`data/track3_obf.tar.gz`が提供されている。

```bash
tar -zxvf ./data/track3_obf.tar.gz -C ./data/
```

データを集約

```bash
python ./track3/dataset/main/formatter.py  --dataset_dir ./data/track3_obf  --output_dir ./data/mos/track3
```

# Data for the VoiceMOS Challenge 2022

[zendo:Data for the VoiceMOS Challenge 2022](https://zenodo.org/records/6572573)

`data/bvcc.zip`として配置されているとする.

bilzard main trackの処理

```bash
unzip ./data/bvcc.zip -d ./data/bvcc
tar -zxvf ./data/track3_obf.tar.gz -C ./data/
tar -zxvf ./data/bvcc/main.tar.gz -C ./data/bvcc/
tar -zxvf ./data/bvcc/ood.tar.gz -C ./data/bvcc/
tar -zxvf ./data/bvcc/scoring_program_distribute.tar.gz -C ./data/bvcc/

cd ./data/bvcc/main
python ./01_get.py
python ./02_gather.py
cp ./gathered/* ./DATA/wav/
```

bilzard ood trackの
```bash
cd ./data/bvcc/ood
python ./01_get.py
python ./02_gather.py
cp ./gathered/* ./DATA/wav/
```

データを集約

```bash
python ./track3/dataset/bvcc/formatter_main.py --dataset_dir ./data/bvcc/main/ --output_dir ./data/mos/bvccmain

python ./track3/dataset/bvcc/formatter_ood.py --dataset_dir ./data/bvcc/ood/ --output_dir ./data/mos/bvccood
```

# SOMOS

[zenodo:SOMOS](https://zenodo.org/records/7378801)

```bash
unzip ./data/somos.zip -d ./data/somos
unzip ./data/somos/audios.zip -d ./data/somos

# データを集約
python ./track3/dataset/somos/formatter.py --dataset_dir ./data/somos/ --output_dir ./data/mos/somos
```

# 集約したデータセット(./data/mos)

ユーザ情報をまとめる

```bash
python ./track3/dataset/utils/user_gather.py --dataset_dir ./data/mos/
```


# libritts-r

```bash
cd ./data
wget https://www.openslr.org/resources/141/train_clean_360.tar.gz
tar -zxvf train_clean_360.tar.gz
python track3/dataset/librittsr/formatter.py --input_dir ./data/LibriTTS/train-clean-360/ --output_dir ./data/libritts/wav

wget https://www.openslr.org/resources/60/train-clean-360.tar.gz
tar -zxvf train-clean-360.tar.gz
python track3/dataset/librittsr/formatter.py --input_dir ./data/LibriTTS/train-clean-360/ --output_dir ./data/libritts/wav
```