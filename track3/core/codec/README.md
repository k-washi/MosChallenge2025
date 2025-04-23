# Audio Codec

Audio Codecを用いて、劣化音声を作成する

## データセット形式のルール



```txt
オリジナル音源だが、 299 > 199の関係性
/datsetname/299
/datasetname/199

298 > 297 > 296 > 295 > ... の順で劣化(この列は、299に対するrestorationの結果)
288, 278, 276, ...は、それぞれ1桁の数字と同じレベルのrestoration + ノイズを付加したもの

x88は, codecの一部を除去 or 空白が1サンプルから~2割
x78は, codecの一部にランダムなノイズ
x68は, codecの一部にランダムなノイズ + (codecの一部を除去 or 空白が1サンプルから~2割)

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

## EncDec戦略

num_strems = 6 or 5でx89を作成
num_strems = 2でx88を作成
num_strems = 1でx87を作成