#!/bin/bash

python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05000/ckpt/ckpt-1050/model.ckpt --test_csv /data/mosranking/track3/fold_0.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05001/ckpt/ckpt-450/model.ckpt --test_csv /data/mosranking/track3/fold_1.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05002/ckpt/ckpt-900/model.ckpt --test_csv /data/mosranking/track3/fold_2.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05003/ckpt/ckpt-1950/model.ckpt --test_csv /data/mosranking/track3/fold_3.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05004/ckpt/ckpt-1500/model.ckpt --test_csv /data/mosranking/track3/fold_4.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05005/ckpt/ckpt-3450/model.ckpt --test_csv /data/mosranking/track3/fold_5.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05006/ckpt/ckpt-3000/model.ckpt --test_csv /data/mosranking/track3/fold_6.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05007/ckpt/ckpt-4050/model.ckpt --test_csv /data/mosranking/track3/fold_7.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05008/ckpt/ckpt-1650/model.ckpt --test_csv /data/mosranking/track3/fold_8.csv --output_dir ./data/i5000
python ./track3/exp/inference/i5000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v05009/ckpt/ckpt-1500/model.ckpt --test_csv /data/mosranking/track3/fold_9.csv --output_dir ./data/i5000