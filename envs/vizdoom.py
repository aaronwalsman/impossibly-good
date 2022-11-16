import gym
from gym.envs.registration import register

from envs.env_wrappers import (
    ProcessFrame,
    EnvSwitcher,
    Waypointer,
    WaypointerVertex,
)

#def wrap_doom_env(env_name):
#    env = gym.make(env_name)
#    wrapped_env = ProcessFrame(env, 84, 84)
#    
#    return wrapped_env

#def wrapped_test():
#    return wrap_doom_env('VizdoomImpossiblyGoodTest-v0')

def monster_room_a_vertices():
    vertex_a = WaypointerVertex([0,-32])
    vertex_b = WaypointerVertex([0,32])
    vertex_c = WaypointerVertex([0,64])
    vertex_d = WaypointerVertex([-30,128])
    vertex_e = WaypointerVertex([0,192])
    
    vertex_f = WaypointerVertex([64,128], use=True)
    
    vertex_g = WaypointerVertex([96,128])
    vertex_h = WaypointerVertex([160,128])
    
    vertex_i = WaypointerVertex([224,128], use=True)
    vertex_j = WaypointerVertex([324,128])
    
    vertex_k = WaypointerVertex([160,-32])
    vertex_l = WaypointerVertex([-160,-32])
    
    # edges
    vertex_a.target = vertex_b
    vertex_b.target = vertex_f
    vertex_c.target = vertex_f
    vertex_d.target = vertex_f
    vertex_e.target = vertex_f
    
    vertex_f.target = vertex_g
    vertex_g.target = vertex_h
    vertex_h.target = vertex_i
    vertex_i.target = vertex_j
    
    vertex_k.target = vertex_a
    vertex_l.target = vertex_a
    
    return [
        vertex_a,
        vertex_b,
        vertex_c,
        vertex_d,
        vertex_e,
        vertex_f,
        vertex_g,
        vertex_h,
        vertex_i,
        vertex_j,
        vertex_k,
        vertex_l
    ]

def monster_room_a_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomA-v0')
    env = Waypointer(env, vertices=monster_room_a_vertices())
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

def monster_room_a_easy_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomAEasy-v0')
    env = Waypointer(env, vertices=monster_room_a_vertices())
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

def monster_room_b_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomB-v0')
    vertices = monster_room_a_vertices()
    for v in vertices:
        v.p[0] = v.p[0] * -1
    env = Waypointer(env, vertices=vertices)
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

def monster_room_b_easy_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomBEasy-v0')
    vertices = monster_room_a_vertices()
    for v in vertices:
        v.p[0] = v.p[0] * -1
    env = Waypointer(env, vertices=vertices)
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

def monster_room():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterRoomAExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterRoomBExpert-v0')
    return EnvSwitcher(enva, envb)

def monster_room_easy():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterRoomAEasyExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterRoomBEasyExpert-v0')
    return EnvSwitcher(enva, envb)

def register_vizdoom_envs():
    register(
        id='ImpossiblyGoodVizdoomTest-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'test_01.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomA-v0',
        #entry_point='envs.vizdoom:monster_room_a_expert',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomAEasy-v0',
        #entry_point='envs.vizdoom:monster_room_a_expert',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_a_easy.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomAExpert-v0',
        entry_point='envs.vizdoom:monster_room_a_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomAEasyExpert-v0',
        entry_point='envs.vizdoom:monster_room_a_easy_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
    #register(
    #    id='ImpossiblyGoodVizDoomMonsterRoomAProcessExpert-v0',
    #    entry_point='envs.vizdoom:monster_room_a_expert',
    #    kwargs={'process':True},
    #)
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomBExpert-v0',
        entry_point='envs.vizdoom:monster_room_b_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomBEasyExpert-v0',
        entry_point='envs.vizdoom:monster_room_b_easy_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
    #register(
    #    id='ImpossiblyGoodVizDoomMonsterRoomBProcessExpert-v0',
    #    entry_point='envs.vizdoom:monster_room_b_expert',
    #    kwargs={'process':True},
    #)
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomB-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_b.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomBEasy-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_b_easy.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoom-v0',
        entry_point='envs.vizdoom:monster_room',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomEasy-v0',
        entry_point='envs.vizdoom:monster_room_easy',
    )
