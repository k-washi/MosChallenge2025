# Audio Codec

Audio Codecを用いて、劣化音声を作成する

## データセット形式のルール



```txt
オリジナル音源だが、 299 > 199の関係性
/datsetname/299
/datasetname/199

x88はnum_streams = 6で劣化
x87はnum_streams = 2で劣化
x86はnum_streams = 1で劣化

x78は、nvidia/low-frame-rate-speech-codec-22khzで劣化
x77は、nvidia/low-frame-rate-speech-codec-22khzで劣化 + codecをmax 9/10でclamp
x76は、nvidia/low-frame-rate-speech-codec-22khzで劣化 + codecをmax 7/10でclamp 

ここで、x88とx78, x88とx87は比較できない
```

以下のディレクトリ配下にwavファイルを配置する

```txt
4.3G    /data/mosranking/somos/199
87M     /data/mosranking/track3/199
420M    /data/mosranking/bvccodd/199
32G     /data/mosranking/libritts/199 # libritts
31G     /data/mosranking/libritts/299 # libritts-r
1.2G    /data/mosranking/bvccmain/199
68G     /data/mosranking/
```

flac 16khzに変換する

```bash
python ./track3/core/codec/to_flac16k.py 
```

```txt
2.0G    /data/mosranking/somos/199
34M     /data/mosranking/track3/199
112M    /data/mosranking/bvccodd/199
13G     /data/mosranking/libritts/199
13G     /data/mosranking/libritts/299
520M    /data/mosranking/bvccmain/199
29G     /data/mosranking/
```


# [efficient-speech-codec](https://github.com/yzGuu830/efficient-speech-codec)

## 依存関係のインストール

```bash
git clone https://github.com/yzGuu830/efficient-speech-codec.git ./libs/efficient-speech-codec

mkdir pretrained
gdown https://drive.google.com/uc?id=180Q4zctqeNnDmRvoMsVQ-3iCB5FriJbN -O ./pretrained/
unzip ./pretrained/esc_large_non_adv.zip -d ./pretrained/
```

```bash
echo "export PYTHONPATH="$PWD/libs/efficient-speech-codec:$PYTHONPATH"" >>  ~/.bashrc

source ~/.bashrc
```