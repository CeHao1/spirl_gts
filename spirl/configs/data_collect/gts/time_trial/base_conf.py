
import imp
import os
from turtle import done

from spirl.utils.general_utils import AttrDict
from spirl.utils.gts_utils import chosen_feature_keys, action_keys

from spirl.gts_demo_sampler.init.base_init import BaseInit
from spirl.gts_demo_sampler.start.base_start import BaseStart
from spirl.gts_demo_sampler.done.base_done import BaseDone
from spirl.gts_demo_sampler.sample.base_sample import BaseSample
from spirl.gts_demo_sampler.file.base_file import BaseFile

# ip_address = '192.168.1.5',
do_init = False
# do_init = True

# configs to initialize the gts
init_config = AttrDict(
    ip_address = '192.168.1.5',
    num_cars = 2,
    bop = [[1, 1], [0.8, 1.2]]
)


# configs to formulate start_condition
start_config = AttrDict(
    num_cars = init_config.num_cars,
    course_v = [1200, 1240],
    speed_kmph = [144, 144],
)

# config for the done function
done_config = AttrDict(
    max_course_v = 2400,
    # max_time = 40,
    # max_lap_count = 2,
)

# config for the sampler
sample_config = AttrDict(
    ip_address = init_config.ip_address,
    min_frames_per_action = 6,
    builtin_controlled = [0, 1],
)

# configs to save and convert the 
file_config = AttrDict(
    save_car_id = [0]
)


# configuration of each module
configuration = AttrDict(
    init = BaseInit,
    start = BaseStart,
    done = BaseDone,
    sample = BaseSample,
    file = BaseFile,

    do_init = do_init,
    start_num_epoch = 2,
    num_epochs = 2,
)