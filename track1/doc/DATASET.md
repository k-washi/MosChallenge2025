# MusicEval Phase1 Datasetのダウンロード
```
gdown https://drive.google.com/uc?id=1KLdpZTgscdMJz0qBUDqnVEKExJzUG5U7 -O ./data/
```

`./data/MusicEval-phase1/`にデータを格納することとする。


# 特徴量の作成

Aux featureの作成

```bash
wget https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth -P ./pretrained/
```

```bash
python ./track1/core/dataset/create_features.py --dataset_dir data/MusicEval-phase1 --output_dir data/MusicEval-phase1/feat
```

## データセットを48kHzに拡張する場合

[AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution?tab=readme-ov-file)

```
uv pip install audiosr
```

```bash
ls ./data/MusicEval-phase1/wav/ > ./data/MusicEval-phase1/wav_list.lst
cd data/MusicEval-phase1/wav
mkdir ../wav48k_audiosr
audiosr -il ../wav_list.lst -s ../wav48k_audiosr --model_name basic --ddim_steps 200
```
