
import os

from spirl.utils.general_utils import AttrDict

from spirl.gts_demo_sampler.init.base_init import BaseInit
# from spirl.gts_demo_sampler.start.base_start import BaseStart
from spirl.gts_demo_sampler.start.pos_start import PosStart_by_course_v
from spirl.gts_demo_sampler.done.base_done import BaseDone
from spirl.gts_demo_sampler.sample.base_sample import BaseSample
from spirl.gts_demo_sampler.file.base_file import BaseFile

do_init = True

# configs to initialize the gts
init_config = AttrDict(
    ip_address = '192.168.1.5',
    num_cars = 1,
    car_name = 'Audi TTCup',
    course_name = 'Tokyo Central Outer',
    tire_type = 'RH',
)

# configs for the start condition
start_config = AttrDict(
    track_dir = os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/track.csv"),
    course_v_range = [1200, 2200],
    speed_kmph_range = [0, 144],
    ey_range_percent = [-0.6, 0.6], # half width
    epsi_range_pi_percent = [-0.1, 0.1] # +- pi/2, positive direction

)

# config for the done function
done_config = AttrDict(
    max_course_v = 2400,
    max_time = 20,
)

# config for the sampler
sample_config = AttrDict(
    ip_address = init_config.ip_address,
    min_frames_per_action = 6,
    builtin_controlled = [0],
    # store_states = True,
)

# configuration of each module
configuration = AttrDict(
    init = BaseInit,
    start = PosStart_by_course_v,
    done = BaseDone,
    sample = BaseSample,
    file = BaseFile,

    do_init = do_init,
    start_num_epoch = 0,
    num_epochs = 3000,
)