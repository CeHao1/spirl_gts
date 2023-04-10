


export EXP_DIR=./experiments
export DATA_DIR=./data

IP:
'PS4-1' : '192.168.1.125', 
'PS4-2' : '192.168.1.119',
'PS5-11': '192.168.1.124', 
'PS5-12': '192.168.1.120',
'PS5-13': '192.168.1.127', 
'PS5-14': '192.168.1.118', 
'PS5-15': '192.168.1.121', 
'PS5-16': '192.168.1.126',
'PS5-17': '192.168.1.116', 
'PS5-18': '192.168.1.123', 


# sample data
python spirl/gts_demo_sampler/sample_demo.py \
    --path spirl/configs/data_collect/gts/time_trial/c2 \
    --ip_address '192.168.1.125' \
    --prefix 'batch_0'


# learn skill

python3 spirl/train.py --path=spirl/configs/skill_prior_learning/gts_corner2/hierarchical_cd --val_data_size=160 \
--gpu=0 --prefix=test_gtsdataset_02


# train RL
python3 spirl/rl/train.py --path=spirl/configs/rl/gts_corner2/SAC_single --seed=0 --gpu=0 \
--prefix=sac_test_01