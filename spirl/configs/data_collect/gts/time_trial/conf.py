
import os

from spirl.utils.general_utils import AttrDict

from spirl.gts_demo_sampler.init.base_init import BaseInit
from spirl.gts_demo_sampler.start.base_start import BaseStart


ip_address = '192.169.1.100',


# configs to initialize the gts
init_config = AttrDict(
    ip_address = ip_address,

)


# configs to formulate start_condition
start_config = AttrDict(

)

# config for the done function
done_config = AttrDict(
    max_course_v = 2000,
    max_time = 100,
)

# config for the sampler
sample_config = AttrDict(


)

# configs to save and convert the 
file_config = AttrDict(

)


# configuration of each module
configuration = AttrDict(
    init = BaseInit,
    start = BaseStart,

)