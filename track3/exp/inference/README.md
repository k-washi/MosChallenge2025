
# 3000番台推論

```
bash ./track3/exp/inference/i3000.sh 
python ./track3/exp/inference/gather_inference.py --input_dir ./data/i3000/ --output_txt ./data/i3000.txt
```

# 4000番台推論

```
bash ./track3/exp/inference/i4000.sh
python ./track3/exp/inference/gather_inference.py --input_dir ./data/i4000/ --output_txt ./data/i4000.txt
```

# 5000番台推論

```
bash ./track3/exp/inference/i5000.sh
python ./track3/exp/inference/gather_inference.py --input_dir ./data/i5000/ --output_txt ./data/i5000.txt
```

```
python ./track3/exp/oof/lgbm_fold_and_eval.py
python ./track3/exp/inference/gather_inference.py --input_dir ./data/oof --output_txt ./data/oof.txt
zip -j oof.zip answer.txt 
```