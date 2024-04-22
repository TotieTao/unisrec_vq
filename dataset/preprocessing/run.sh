#!/bin/bash
for ds in Games
do
  python v2_process_amazon.py --dataset ${ds} --output_path ../pretrain/ --word_drop_ratio 0.15
done

python v2_to_pretrain_atomic_files.py --datasets Games

path=`pwd`
for ds in Games
do
  ln -s ${path}/../pretrain/${ds}/${ds}.feat1CLS ../pretrain/G/
  ln -s ${path}/../pretrain/${ds}/${ds}.feat2CLS ../pretrain/G/

done