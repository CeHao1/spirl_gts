
import numpy as np
import os
import gym
import pandas as pd
from spirl.utils.general_utils import ParamDict, AttrDict

#  =========================== env setup utils ================================
CAR_CODE = {'Mazda Roadster':   2148, 
            'Mazda Demio':      3383, 
            'Audi TTCup':       3298
            }

COURSE_CODE = { 'Tokyo Central Outer':      351, 
                'Tokyo East Outer':         361, 
                'Tokyo Central Inner':      450, 
                'Brandas Hatch':            119,
                'protect':                  452
            }

BOP     =   {
            'default':      {"enable": False, "power": 100, "weight": 100},
            'Mazda Demio':  {"enable": True, "power": 124, "weight": 119},
            'Audi TTCup' :  {"enable": True, "power": 104, "weight": 97},
            }

# tire type: 
# CH, CM, CS, (comfort)
# SH, SM, SS, (sport)
# RH, RM, RS, RSS, (racing hard)
# IM, HW, or DIRT (wet tire)

TIRE_TYPE = 'RH'

RL_OBS_1 = ['lap_count', 'current_lap_time_msec', 'speed_kmph', 'frame_count', 'is_controllable',
                'vx', 'vy', 'vz', 'pos','rot', 'angular_velocity', 'front_g', 'side_g', 'vertical_g',
                'centerline_diff_angle', 'centerline_distance', 'edge_l_distance', 'edge_r_distance', 'course_v',
                'is_hit_wall', 'is_hit_cars', 'hit_wall_time', 'hit_cars_time',
                'steering', 'throttle', 'brake'] + \
                ["curvature_in_%.1f_%s" % (step, "seconds") for step in np.arange(start=0.2, stop=3.0, step=0.2)] \
                + ["lidar_distance_%i_deg" % deg for deg in np.concatenate(
                (np.arange(start=0, stop=105, step=15), np.arange(start=270, stop=360, step=15),))]


ego_obs = ['Vx', 'Vy', 'Vz', 'dpsi' , 'ax', 'ay', 'az', 'epsi', 'ey', 'Wl', 'Wr'] + \
              ['hit_wall', 'hit_car', 'delta' , 'thr', 'brk']
mode = "seconds"
seconds_into_future = np.arange(start=0.2, stop=3.0, step=0.2)
curvature_name_space = ["curvature_in_%.1f_%s" % (step, mode) for step in seconds_into_future]
lidar_name_space = ["lidar_distance_%i_deg" % deg for deg in np.concatenate(
            (np.arange(start=0, stop=105, step=15), np.arange(start=270, stop=360, step=15),))]

ego_obs += curvature_name_space
ego_obs += lidar_name_space


DEFAULT_FEATURE_KEYS = (
    [
        "front_g",
        "side_g",
        "vertical_g",
        "vx",
        "vy",
        "vz",
        "centerline_diff_angle",
        "is_hit_wall",
        "steering",
    ]
    + [
        "lidar_distance_%i_deg" % deg
        for deg in np.concatenate(
            (
                np.arange(start=0, stop=105, step=15),
                np.arange(start=270, stop=360, step=15),
            )
        )
    ]
    # + [
    #     "curvature_in_%.1f_seconds" % seconds
    #     for seconds in np.arange(start=1, stop=3.0, step=0.2)
    # ]

    +[
        "curvature_in_%.1f_seconds" % seconds
        for seconds in np.arange(start=0.2, stop=3.0, step=0.2)
    ]
)

chosen_feature_keys = DEFAULT_FEATURE_KEYS
action_keys = ['steering', 'throttle-brake']
# chosen_feature_keys = ego_obs

# state_dim = len(ego_obs)
state_dim = len(chosen_feature_keys)

def obs2name(obs):
    state = {}
    kap = []
    lidar = []
    for i in range(len(ego_obs)):
        if 'curvature_in_' in ego_obs[i]:
            kap.append(obs[i])
        elif 'lidar_distance_' in ego_obs[i]:
            lidar.append(obs[i])
        else:
            state[ego_obs[i]] = obs[i]
    state['kap'] = kap
    state['lidar'] = lidar
    return state

def start_condition_formulator(num_cars, course_v, speed):
    conditions = []
    for car_id in range(num_cars):
        conditions.append(
            {
                "id": car_id,
                "course_v": course_v[car_id],
                "speed_kmph": speed[car_id],
                "angular_velocity": 0,
            }
        )

    start_conditions = {"launch": conditions}
    return start_conditions


def BoP_formulator(num_cars, car_name, weight_percentage, power_percentage):
    bop_base = BOP[car_name]
    power_base = bop_base['power']
    weight_base = bop_base['weight']

    bops = []
    for car_id in range(num_cars):
        bops.append( {"enable": True, 
                        "power": round(power_base * power_percentage[car_id]), 
                        "weight": round(weight_base * weight_percentage[car_id]) } )

    return bops

def initialize_gts(ip, num_cars, car_codes, course_code, tire_type, bops):
    # print(ip, num_cars, car_codes, course_code, tire_type, bops)

    from gym_gts import GTSApi
    with GTSApi(ip=ip) as gts_api:
        gts_api.set_race(
            num_cars = num_cars,
            car_codes = car_codes,
            course_code = course_code,
            front_tires = tire_type,
            rear_tires = tire_type,
            bops = bops
        )


def make_env(**kwarg):
    env = gym.make('gts-v0', **kwarg)
    return env


#  =================================== state observe ===============================
def WrapToPi(x):
    x = np.mod(x, 2*np.pi)
    x -= 2*np.pi * np.heaviside((x - np.pi), 0)
    return x

def add_dim(l):
    if l.ndim == 1:
        return l[None]
    else:
        return l

def bool_2_int(l):
    return np.array([1 if x else 0 for x in l]).squeeze()

def convert_simple_states(gts_state):
    state = {}

    for key in gts_state:
        if not isinstance(gts_state[key], np.ndarray): 
            gts_state[key] = np.array(gts_state[key])

        # car gts frame states 
        if key == 'current_lap_time_msec':
            state['t'] = gts_state[key]/1000
        elif key == 'speed_kmph':
            state['V'] = gts_state[key] / 3.6
        elif key == 'steering':
            state['delta'] = gts_state[key]
        elif key == 'throttle':
            state['thr'] = gts_state[key]
            state['thr-brk'] = gts_state['throttle'] - gts_state['brake']
        elif key == 'brake':
            state['brk'] = gts_state[key]

         # car states in frenet coordinate
        elif key == 'centerline_diff_angle':
            state['epsi'] = gts_state[key]
        elif key == 'centerline_distance':
            state['ey'] = - gts_state[key]
        elif key == 'course_v':
            state['s'] = gts_state['course_v']
        elif key == 'frame_count':
            state['frame_count'] = gts_state[key]
        elif key == 'lap_count':
            state['lap_count'] = gts_state['lap_count']

        elif key == 'edge_l_distance':
            state['Wl'] = gts_state['edge_l_distance']
        elif key == 'edge_r_distance':
            state['Wr'] = gts_state['edge_r_distance']

        elif key == 'pos[0]':
            state['X'] = gts_state[key]
        elif key == 'pos[2]':
            state['Y'] = gts_state[key]
        elif key == 'pos[1]':
            state['Z'] = gts_state[key]
        elif key == 'rot[0]': # pitch yaw roll
            state['Theta'] = - gts_state[key]
        elif key == 'rot[1]': # pitch yaw roll
            state['Psi'] = WrapToPi( np.pi/2 - gts_state[key] )
        elif key == 'rot[2]': # pitch yaw roll
            state['Phi'] = gts_state[key]
        elif key == 'vx':
            state['Vx'] = gts_state['vx']
        elif key == 'vy':
            state['Vy'] = gts_state['vy']
        elif key == 'vz':
            state['Vz'] = gts_state['vz']
        elif key == 'angular_velocity[1]':
            state['dpsi'] = - gts_state[key]
        elif key == 'angular_velocity[0]':
            state['dtheta'] = - gts_state[key]
        elif key == 'angular_velocity[2]':
            state['dphi'] = gts_state[key]
        elif key == 'front_g':
            state['ax'] = gts_state['front_g']
        elif key == 'side_g':
            state['ay'] = - gts_state['side_g']
        elif key == 'vertical_g':
            state['az'] = gts_state['vertical_g']

         # hit
        elif key == 'is_hit_wall':
            state['hit_wall'] = bool_2_int(gts_state[key])
        elif key == 'is_hit_cars':
            state['hit_car'] = bool_2_int(gts_state[key])
        elif key == 'hit_wall_time':
            state['hit_wall_time'] = gts_state['hit_wall_time']
        elif key == 'hit_cars_time':
            state['hit_cars_time'] = gts_state['hit_cars_time']
        elif key == 'shift_position':
            state['shift'] = gts_state['shift_position']
        elif key == 'is_controllable':
            state['controllable'] = bool_2_int(gts_state[key])
        elif 'curvature' in key:
            state[key] = gts_state[key]
        elif 'lidar' in key:
            state[key] = gts_state[key]

        else:
            pass

    return state

def gts_state_2_cartesian(gts_state):
    state = {}
    # to store rotation at first
    if 'rot' in gts_state:
        state.update(gts_state_2_cartesian( [{'rot':gts_state['rot']}] ) )
       
    for key in gts_state:
        if not isinstance(gts_state[key], np.ndarray): 
            gts_state[key] = np.array(gts_state[key])

        # car gts frame states 
        if key == 'current_lap_time_msec':
            state['t'] = gts_state[key]/1000
        elif key == 'speed_kmph':
            state['V'] = gts_state[key] / 3.6
        elif key == 'steering':
            state['delta'] = gts_state[key]
        elif key == 'throttle':
            state['thr'] = gts_state[key]
        elif key == 'brake':
            state['brk'] = gts_state[key]

        # car states in frenet coordinate
        elif key == 'centerline_diff_angle':
            state['epsi'] = gts_state[key]
        elif key == 'centerline_distance':
            state['ey'] = - gts_state[key]
        elif key == 'course_v':
            state['s'] = gts_state['course_v']
        elif key == 'frame_count':
            state['frame_count'] = gts_state[key]
        elif key == 'lap_count':
            state['lap_count'] = gts_state['lap_count']

        elif key == 'edge_l_distance':
            state['Wl'] = gts_state['edge_l_distance']
        elif key == 'edge_r_distance':
            state['Wr'] = gts_state['edge_r_distance']

        # car chassis state in cartesian coordinate
        elif key == 'pos':
            temp_pos = add_dim(gts_state[key])
            state['X'] = temp_pos[:, 0].squeeze()
            state['Y'] = temp_pos[:, 2].squeeze()
            state['Z'] = temp_pos[:, 1].squeeze()
        elif key == 'rot': # pitch yaw roll
            temp_angle = add_dim(gts_state[key])
            state['Theta'] = - temp_angle[:, 0].squeeze()
            state['Psi'] = WrapToPi( np.pi/2 - temp_angle[:, 1].squeeze() )
            state['Phi'] = temp_angle[:, 2].squeeze()
        elif key == 'vx':
            state['Vx'] = gts_state['vx']
        elif key == 'vy':
            state['Vy'] = gts_state['vy']
        elif key == 'vz':
            state['Vz'] = gts_state['vz']
        elif key == 'angular_velocity':
            # roll, pitch, yaw
            temp_angle = add_dim(gts_state[key])
            state['dpsi'] = - temp_angle[:, 1].squeeze()
            state['dtheta'] = - temp_angle[:, 0].squeeze()
            state['dphi'] = temp_angle[:, 2].squeeze()
        elif key == 'front_g':
            state['ax'] = gts_state['front_g']
        elif key == 'side_g':
            state['ay'] = - gts_state['side_g']
        elif key == 'vertical_g':
            state['az'] = gts_state['vertical_g']

        # car tire states
        # 0 front left, 1 front right, 2 rear left, 3 rear right
        elif key == 'wheel_load':
            value = add_dim(gts_state[key])
            state['Fz4'] = value
            state['Fzf'] = np.sum(value[:, 0:2], axis=0).squeeze()
            state['Fzr'] = np.sum(value[:, 2:4], axis=0).squeeze()
            state['Fz'] = np.sum(value, axis=0).squeeze()
        elif key == 'slip_angle':
            value = add_dim(gts_state[key])
            state['alpha4'] = value
            state['alphaf'] = np.mean(value[:, 0:2], axis=0).squeeze()
            state['alphar'] = np.mean(value[:, 2:4], axis=0).squeeze()
        elif key == 'slip_ratio':
            value = add_dim(gts_state[key])
            state['sigma'] = value
            state['sigmaf'] = np.mean(value[:, 0:2], axis=0).squeeze()
            state['sigmar'] = np.mean(value[:, 2:4], axis=0).squeeze()
        elif key == 'wheel_angle':
            value = add_dim(gts_state[key])
            state['delta4'] = value
            state['deltaf'] = np.mean(value[:, 0:2], axis=0).squeeze()
            state['deltar'] = np.mean(value[:, 2:4], axis=0).squeeze()
        elif key == 'wheel_omega':
            value = add_dim(gts_state[key])
            state['omega4'] = value
            state['omegaf'] = np.mean(value[:, 0:2], axis=0).squeeze()
            state['omegar'] = np.mean(value[:, 2:4], axis=0).squeeze()

        # engine states
        elif key == 'engine_torque':
            state['Te'] = gts_state['engine_torque']

        # hit
        elif key == 'is_hit_wall':
            state['hit_wall'] = bool_2_int(gts_state[key])
        elif key == 'is_hit_cars':
            state['hit_car'] = bool_2_int(gts_state[key])
        elif key == 'hit_wall_time':
            state['hit_wall_time'] = gts_state['hit_wall_time']
        elif key == 'hit_cars_time':
            state['hit_cars_time'] = gts_state['hit_cars_time']
        elif key == 'shift_position':
            state['shift'] = gts_state['shift_position']
        elif key == 'is_controllable':
            state['controllable'] = bool_2_int(gts_state[key])
        elif 'curvature' in key:
            state[key] = gts_state[key]
        elif 'lidar' in key:
            state[key] = gts_state[key]

        else:
            pass

    return state

def raw_observation_to_true_observation(raw_obs):
    gts_states = gts_observation_2_state(raw_obs, RL_OBS_1)
    states = gts_state_2_cartesian(gts_states)
    return states_2_obs(states)

def gts_observation_2_state(observation, feature_keys):
    gts_state = {}
    for obs,key in zip(observation, feature_keys):
        gts_state[key] = np.array(obs)

    return gts_state

def states_2_obs(states):
    observation = []
    for key in ego_obs:
        observation.append(states[key])
    return observation

#  ======================== offline file opeartions =========================
def clip_dict(states, idx):
    for name in states:
        states[name] = states[name][idx]

def clip_states(states, idx):
    state_array = {}
    for name in states:
        if np.array(states[name]).ndim == 1:
            state_array[name] = states[name][idx]
        else:
            temp_list = []
            for idx_state in range(len(states[name])):
                temp_list.append( states[name][idx_state][idx] )
            state_array[name] = temp_list
    return state_array


def convert_coordinates(state):
    # for state in states:
    state["pos[2]"] *= -1
    state["velocity[2]"] *= -1
    state["rot[0]"] *= -1
    state["rot[1]"] *= -1
    state["angular_velocity[0]"] *= -1
    state["angular_velocity[1]"] *= -1
    return state

def velocities_to_car_oriented(state):
    # for state in states:
    vx = state["velocity[0]"]
    vy = state["velocity[2]"]
    # transform to counter clockwise angle starting at x axis
    theta = np.pi / 2 - (state["rot[1]"])

    # apply clockwise rotation
    state["vx"] = vx * np.cos(theta) + vy * np.sin(theta)
    state["vy"] = -vx * np.sin(theta) + vy * np.cos(theta)
    state["vz"] = state["velocity[1]"]

    return state
   

def load_replay_2_states(file_dir, file_name, car_key='car0', chosen_lap=None, method='h5'):
    os.path.join(file_dir, file_name)
    data_dir = os.path.join(file_dir, file_name)

    if method == 'h5':
        data = pd.read_hdf(data_dir, key=car_key, mode="r")
    elif method == 'csv':
        data = pd.read_csv(data_dir)
        data = convert_coordinates(data)
        data = velocities_to_car_oriented(data)

    data_np = {}
    for name in data:
        data_np[name] = data[name].to_numpy()

    states = convert_simple_states(data_np)
    # print(states.keys())
    if chosen_lap is not None:
        idx = np.where(states['lap_count']==chosen_lap)
        clip_dict(states, idx)
    return states

def load_track(track_dir):
    data = pd.read_csv(track_dir)
    data2 = {}
    for d in data:
        data2[d] = data[d].values
    track = AttrDict(data2)
    return track

#=================================================================================

def load_standard_table():
    
    import os
    # from sklearn.preprocessing import StandardScaler
    import pickle
    try:
        file_path = os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/gts/standard_table")
        f = open(file_path, "rb")
        standard_table = pickle.load(f)
        f.close()

        state_scaler = standard_table['state']
        action_scaler = standard_table['action']

        print("load standard table successful")
        return state_scaler, action_scaler
    except:
        print("not standard table")

# ======================================== reward function =====================================

maf = 6
c_wall_hit = 1/(2000*10/9.3)
horizon = 100
max_eval_lap = 100

def time_done(seconds, state):
    """ Determines if game time of 'state' is bigger than 'seconds' """
    return state["frame_count"] > (60 * seconds) if state else False


def sampling_done_function(state):
    return time_done(horizon, state)


def evaluation_done_function(state):
    return state["lap_count"] > 2 or state["current_lap_time_msec"]/1000.0 > max_eval_lap if state else False

def eval_time_trial_done_function(state):
    return state["lap_count"] > 3 and state["current_lap_time_msec"]/1000.0 > 5.0 if state else False

def reward_function(state, previous_state, course_length):
    if previous_state \
            and isinstance(previous_state["course_v"], float) \
            and isinstance(previous_state["lap_count"], int):

        # version robust to step length through scaling and always detecting wall contact (other than is_hit_wall)
        reward = (
                         - (
                                 (state["hit_wall_time"] - previous_state["hit_wall_time"])
                                 * 10 * state["speed_kmph"]**2 * c_wall_hit)
                         + (state["course_v"] + state["lap_count"] * course_length)
                         - (previous_state["course_v"] + previous_state["lap_count"] * course_length)
                 ) * (maf/(state["frame_count"] - previous_state["frame_count"]))  # correcting too long steps

        return reward

def eval_time_trial_reward_function(state, previous_state, course_length):
    if previous_state \
            and isinstance(previous_state["course_v"], float) \
            and isinstance(previous_state["lap_count"], int):

            if (previous_state["lap_count"] == 2 and state["lap_count"] == 3):
                last_t = previous_state["current_lap_time_msec"]/1000.0
                now_t = state["current_lap_time_msec"]/1000.0

                print('now is the second lap time, ', last_t, now_t)

                return max(last_t, now_t)
            else:
                return 0

# =========================== for corner 2 and versus ========================


def corner2_done_function(state):
    # course > 2400 or time > 60 seconds
    return state['course_v'] >= 2400 or state['frame_count'] > 60 * 60

def single_reward_function(state, previous_state, course_length):
    if previous_state \
            and isinstance(previous_state["course_v"], float) \
            and isinstance(previous_state["lap_count"], int):

        # version robust to step length through scaling and always detecting wall contact (other than is_hit_wall)
        reward = (
                         - (
                                 (state["hit_wall_time"] - previous_state["hit_wall_time"])
                                 * 10 * state["speed_kmph"]**2 * c_wall_hit)
                         + (state["course_v"] + state["lap_count"] * course_length)
                         - (previous_state["course_v"] + previous_state["lap_count"] * course_length)
                 ) * (maf/(state["frame_count"] - previous_state["frame_count"]))  # correcting too long steps

        return reward

