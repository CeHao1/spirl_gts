python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM999_s0_02 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM999_s1_01 --seed=1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM999_s2_01 --seed=2

# 999
python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM9999_s0_01 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM9999_s1_01 --seed=1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl3_HYB_GAM9999_s2_01 --seed=2