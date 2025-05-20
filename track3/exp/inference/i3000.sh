#!/bin/bash

python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03000/ckpt/ckpt-600/model.ckpt --test_csv /data/mosranking/track3/fold_0.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03001/ckpt/ckpt-600/model.ckpt --test_csv /data/mosranking/track3/fold_1.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03002/ckpt/ckpt-2100/model.ckpt --test_csv /data/mosranking/track3/fold_2.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03003/ckpt/ckpt-300/model.ckpt --test_csv /data/mosranking/track3/fold_3.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03004/ckpt/ckpt-900/model.ckpt --test_csv /data/mosranking/track3/fold_4.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03005/ckpt/ckpt-2100/model.ckpt --test_csv /data/mosranking/track3/fold_5.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03006/ckpt/ckpt-2100/model.ckpt --test_csv /data/mosranking/track3/fold_6.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03007/ckpt/ckpt-1350/model.ckpt --test_csv /data/mosranking/track3/fold_7.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03008/ckpt/ckpt-600/model.ckpt --test_csv /data/mosranking/track3/fold_8.csv --output_dir ./data/i3000
python ./track3/exp/inference/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03009/ckpt/ckpt-1500/model.ckpt --test_csv /data/mosranking/track3/fold_9.csv --output_dir ./data/i3000