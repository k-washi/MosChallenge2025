#!/bin/bash

python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04000/ckpt/ckpt-1500/model.ckpt --test_csv /data/mosranking/track3/fold_0.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04001/ckpt/ckpt-750/model.ckpt --test_csv /data/mosranking/track3/fold_1.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04002/ckpt/ckpt-1350/model.ckpt --test_csv /data/mosranking/track3/fold_2.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04003/ckpt/ckpt-600/model.ckpt --test_csv /data/mosranking/track3/fold_3.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04004/ckpt/ckpt-750/model.ckpt --test_csv /data/mosranking/track3/fold_4.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04005/ckpt/ckpt-3750/model.ckpt --test_csv /data/mosranking/track3/fold_5.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04006/ckpt/ckpt-2700/model.ckpt --test_csv /data/mosranking/track3/fold_6.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04007_1/ckpt/ckpt-450/model.ckpt --test_csv /data/mosranking/track3/fold_7.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04008_1/ckpt/ckpt-2100/model.ckpt --test_csv /data/mosranking/track3/fold_8.csv --output_dir ./data/i4000
python ./track3/exp/inference/i4000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v04009/ckpt/ckpt-900/model.ckpt --test_csv /data/mosranking/track3/fold_9.csv --output_dir ./data/i4000