#!/bin/bash

python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03000/ckpt/ckpt-600/model.ckpt --output_fp ./data/i3000/f0.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03001/ckpt/ckpt-600/model.ckpt --output_fp ./data/i3000/f1.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03002/ckpt/ckpt-2100/model.ckpt --output_fp ./data/i3000/f2.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03003/ckpt/ckpt-300/model.ckpt --output_fp ./data/i3000/f3.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03004/ckpt/ckpt-900/model.ckpt --output_fp ./data/i3000/f4.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03005/ckpt/ckpt-2100/model.ckpt --output_fp ./data/i3000/f5.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03006/ckpt/ckpt-2100/model.ckpt --output_fp ./data/i3000/f6.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03007/ckpt/ckpt-1350/model.ckpt --output_fp ./data/i3000/f7.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03008/ckpt/ckpt-600/model.ckpt --output_fp ./data/i3000/f8.csv
python ./track3/exp/inference_test/i3000.py --ckpt_file ./logs/utmos_sslwavlm_sfds_fold/v03009/ckpt/ckpt-1500/model.ckpt --output_fp ./data/i3000/f9.csv