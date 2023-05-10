
# python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=0 --gpu=0 \
# --prefix=HL_s0_02

# python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=1 --gpu=0 \
# --prefix=HL_s1_01

# python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi --seed=2 --gpu=0 \
# --prefix=HL_s2_01

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi  --gpu=0 \
--seed=4 --prefix=HYB_td80_s2_24 \
--resume='latest' --strict_weight_loading=0 --resume_load_replay_buffer=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi  --gpu=0 \
--seed=3 --prefix=HYB_td80_s3_22 \
--resume='latest' --strict_weight_loading=0 --resume_load_replay_buffer=0 

python3 spirl/rl/train.py --path=spirl/configs/hrl/gts_corner2/sh_multi  --gpu=0 \
--seed=4 --prefix=HYB_td80_s4_23 \
--resume='latest' --strict_weight_loading=0 --resume_load_replay_buffer=0 