


export EXP_DIR=./experiments
export DATA_DIR=./data


# learn skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=test_gtsdataset_02


# train RL
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_single --seed=0 --gpu=0 \
--prefix=sac_test_01