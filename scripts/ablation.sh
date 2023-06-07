
# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_LL_correct_s1_01 --seed=1

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_LL_correct_s2_01 --seed=2

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_HYB_s1_01 --seed=1

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_HYB_s2_01 --seed=2

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_LL_s01_11 --seed=0

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_LL_s1_01 --seed=1

# python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m2  --gpu=0 \
# --resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
# --prefix=abl1_LL_s2_01 --seed=2

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl2_HYB_LLH_s0_02 --seed=0

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl2_HYB_LLH_s1_01 --seed=1

python3 spirl/rl/train.py --path=spirl/configs/hrl/maze/shLL_m1  --gpu=0 \
--resume='latest' --resume_load_replay_buffer=0 --strict_weight_loading=0 \
--prefix=abl2_HYB_LLH_s2_01 --seed=2