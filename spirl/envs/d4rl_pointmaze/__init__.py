from spirl.envs.d4rl_pointmaze.maze_layouts import rand_layout

from gym.envs.registration import register

register(
    id='maze2d-randMaze-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze0-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=0),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze1-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=1),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze42-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=42),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze0S30-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=0, size=30),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze0S40-ac-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=0, size=40),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze1010-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=1, size=10),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)


register(
    id='maze2d-randMaze2020-v0',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec': rand_layout(seed=0, size=20),
        'agent_centric_view': True,
        'reward_type': 'sparse',
        'reset_target': False,
        'ref_min_score': 4.83,
        'ref_max_score': 191.99,
        'dataset_url':'http://maze2d-hardexpv2-sparse.hdf5'
    }
)

