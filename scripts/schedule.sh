#!/bin/bash

# SD2
CUDA_VISIBLE_DEVICES=0,1 python src/train.py trainer.devices=2 data.args.batch_size=16 model.args.optimizer_para.lr=1e-5 task_name=SD2 \
    data.args.num_workers=16

# SD3 M
CUDA_VISIBLE_DEVICES=0,1 python src/train.py trainer.devices=2 data.args.batch_size=16 model.args.optimizer_para.lr=1e-5 task_name=SD3_M \
    data.args.start_step=4 data.args.end_step=24 data.args.num_files=40000 data.args.num_workers=16 data.args.known_ts=[0.142,0.285,0.428,0.571,0.714,0.857]\
    paths.data_dir=/mnt/sharedata/ssd_large/users/guohl/datasets/DiQE/SD3_M_cocotrain_fullcap-numimgs_alltime_perprompt5_pca48 

# SD3 L
CUDA_VISIBLE_DEVICES=0,1 python src/train.py trainer.devices=2 data.args.batch_size=16 model.args.optimizer_para.lr=1e-5 task_name=SD3_L \
    data.args.start_step=4 data.args.end_step=24 data.args.num_files=40000 data.args.num_workers=4 data.args.known_ts=[0.142,0.285,0.428,0.571,0.714,0.857]\
    paths.data_dir=/mnt/sharedata/ssd_large/users/guohl/datasets/DiQE/SD3_L_cocotrain_fullcap-numimgs_alltime_perprompt5_pca48
