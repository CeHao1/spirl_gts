


export EXP_DIR=./experiments
export DATA_DIR=./data

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=cd_gtsc2_01