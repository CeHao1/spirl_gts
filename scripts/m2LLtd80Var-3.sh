# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=LL_td80_Var-3_s0_03 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td80_Var-3_s1_01 --seed=1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=LL_td80_Var-3_s2_01 --seed=2