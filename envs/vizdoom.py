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

def monster_room_small_a_vertices():
    vertex_a = WaypointerVertex([0,160])
    vertex_b = WaypointerVertex([-64,160])
    vertex_c = WaypointerVertex([64,160], use=True)
    vertex_d = WaypointerVertex([0,64])
    vertex_e = WaypointerVertex([128,170])
    vertex_f = WaypointerVertex([224,128], use=True)
    vertex_g = WaypointerVertex([160,64])
    vertex_h = WaypointerVertex([324,128])
    
    vertex_b.target = vertex_a
    vertex_d.target = vertex_a
    vertex_a.target = vertex_c
    vertex_c.target = vertex_e
    vertex_e.target = vertex_f
    vertex_g.target = vertex_f
    vertex_f.target = vertex_h
    
    return [
        vertex_a,
        vertex_b,
        vertex_c,
        vertex_d,
        vertex_e,
        vertex_f,
        vertex_g,
        vertex_h,
    ]

def monster_room_a_hard_vertices():
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

def monster_room_a_vertices():
    vertex_a = WaypointerVertex([0,-32])
    vertex_b = WaypointerVertex([0,32])
    vertex_c = WaypointerVertex([0,64])
    vertex_d = WaypointerVertex([-64,128], bad_door=True)
    vertex_e = WaypointerVertex([0,192])
    
    vertex_f = WaypointerVertex([64,128], use=True)
    
    vertex_g = WaypointerVertex([96,128])
    vertex_h = WaypointerVertex([160,128])
    
    vertex_i = WaypointerVertex([224,128], use=True)
    vertex_j = WaypointerVertex([324,128])
    
    vertex_k = WaypointerVertex([160,-32])
    vertex_l = WaypointerVertex([-160,-32])
    
    vertex_m = WaypointerVertex([32,0])
    vertex_n = WaypointerVertex([-32,0])
    
    # edges
    vertex_a.target = vertex_b
    vertex_b.target = vertex_c
    vertex_c.target = vertex_f
    vertex_d.target = vertex_f
    vertex_e.target = vertex_f
    
    vertex_f.target = vertex_g
    vertex_g.target = vertex_h
    vertex_h.target = vertex_i
    vertex_i.target = vertex_j
    
    vertex_k.target = vertex_a
    vertex_l.target = vertex_a
    
    vertex_m.target = vertex_b
    vertex_n.target = vertex_b
    
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

def monster_hall_long_a_vertices():
    # top room
    vertex_a = WaypointerVertex([-30, 128], bad_door=True)
    vertex_b = WaypointerVertex([  0, 192])
    vertex_c = WaypointerVertex([64, 128], use=True)
    vertex_d = WaypointerVertex([ 0, 64])
    
    # center hall
    vertex_e = WaypointerVertex([ 0, -64])
    vertex_f = WaypointerVertex([ 0, -256])
        
    # side hall
    vertex_g = WaypointerVertex([96, 128])
    vertex_h = WaypointerVertex([160,128])
    vertex_i = WaypointerVertex([160,-164])
    
    # bottom room
    vertex_j = WaypointerVertex([160,-226])
    vertex_k = WaypointerVertex([96, -256])
    vertex_l = WaypointerVertex([224, -256])
    vertex_m = WaypointerVertex([160, -320], use=True)
    vertex_n = WaypointerVertex([160, -400])
    
    vertex_a.target = vertex_c
    vertex_b.target = vertex_c
    vertex_c.target = vertex_g
    vertex_d.target = vertex_c
    
    vertex_e.target = vertex_d
    vertex_f.target = vertex_e
    
    vertex_g.target = vertex_h
    vertex_h.target = vertex_i
    
    vertex_i.target = vertex_j
    vertex_j.target = vertex_m
    vertex_k.target = vertex_m
    vertex_l.target = vertex_m
    vertex_m.target = vertex_n
    
    vertices = [
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
        vertex_l,
        vertex_m,
        vertex_n,
    ]
    
    return vertices


def monster_hall_a_vertices():
    vertex_a = WaypointerVertex([-96,-32])
    vertex_b = WaypointerVertex([128,-32])
    vertex_c = WaypointerVertex([128,32])
    vertex_d = WaypointerVertex([-32,-32])
    
    vertex_e = WaypointerVertex([-96,64], bad_door=True)
    vertex_f = WaypointerVertex([-32,128], use=True)
    vertex_g = WaypointerVertex([-128,128], bad_door=True)
    vertex_h = WaypointerVertex([-96,192], bad_door=True)
    
    vertex_i = WaypointerVertex([96,128])
    vertex_j = WaypointerVertex([128,64])
    vertex_k = WaypointerVertex([128,192])
    vertex_l = WaypointerVertex([192,128], use=True)
    
    vertex_m = WaypointerVertex([292,128])
    
    vertex_a.target = vertex_e
    vertex_b.target = vertex_d
    vertex_c.target = vertex_d
    vertex_d.target = vertex_a
    
    vertex_e.target = vertex_f
    vertex_f.target = vertex_i
    vertex_g.target = vertex_f
    vertex_h.target = vertex_f
    
    vertex_i.target = vertex_l
    vertex_j.target = vertex_l
    vertex_k.target = vertex_l
    vertex_l.target = vertex_m
    
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
        vertex_l,
    ]

def monster_hall_b_vertices():
    vertex_a = WaypointerVertex([-96,-32])
    vertex_b = WaypointerVertex([128,-32])
    vertex_c = WaypointerVertex([128,32])
    vertex_d = WaypointerVertex([-32,-32])
    
    vertex_e = WaypointerVertex([-96,64], bad_door=True)
    vertex_f = WaypointerVertex([-32,128], bad_door=True)
    vertex_g = WaypointerVertex([-160,128], bad_door=True)
    vertex_h = WaypointerVertex([-96,192], use=True)
    
    vertex_i = WaypointerVertex([-96,256])
    vertex_j = WaypointerVertex([-160,288])
    vertex_k = WaypointerVertex([-32,288])
    vertex_l = WaypointerVertex([-96,352], use=True)
    
    vertex_m = WaypointerVertex([-96,452])
    
    vertex_a.target = vertex_e
    vertex_b.target = vertex_d
    vertex_c.target = vertex_d
    vertex_d.target = vertex_a
    
    vertex_e.target = vertex_h
    vertex_f.target = vertex_h
    vertex_g.target = vertex_h
    vertex_h.target = vertex_i
    
    vertex_i.target = vertex_l
    vertex_j.target = vertex_l
    vertex_k.target = vertex_l
    vertex_l.target = vertex_m
    
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
        vertex_l,
    ]


def monster_hall_a_before_vertices():
    # top room
    vertex_a = WaypointerVertex([-30, 128], bad_door=True)
    vertex_b = WaypointerVertex([  0, 192])
    vertex_c = WaypointerVertex([64, 128], use=True)
    vertex_d = WaypointerVertex([ 0, 64])
    
    # center hall
    vertex_e = WaypointerVertex([ 0, -64])
    vertex_f = WaypointerVertex([ 0, -256])
    
    # side room
    vertex_g = WaypointerVertex([128, 128])
    vertex_h = WaypointerVertex([160, 192])
    vertex_i = WaypointerVertex([128, 64])
    vertex_j = WaypointerVertex([224, 128], use=True)
    vertex_k = WaypointerVertex([324, 128])
    
    vertex_l = WaypointerVertex([160,-160])
    vertex_m = WaypointerVertex([160,-256])
    
    '''
    # side hall
    vertex_g = WaypointerVertex([96, 128])
    vertex_h = WaypointerVertex([160,128])
    vertex_i = WaypointerVertex([160,-164])
    
    # bottom room
    vertex_j = WaypointerVertex([160,-226])
    vertex_k = WaypointerVertex([96, -256])
    vertex_l = WaypointerVertex([224, -256])
    vertex_m = WaypointerVertex([160, -320], use=True)
    vertex_n = WaypointerVertex([160, -400])
    '''
    
    vertex_a.target = vertex_c
    vertex_b.target = vertex_c
    vertex_c.target = vertex_g
    vertex_d.target = vertex_c
    
    vertex_e.target = vertex_d
    vertex_f.target = vertex_e
    
    vertex_g.target = vertex_j
    vertex_h.target = vertex_j
    vertex_i.target = vertex_j
    vertex_j.target = vertex_k
    
    vertex_l.target = vertex_i
    vertex_m.target = vertex_l
    
    '''
    vertex_g.target = vertex_h
    vertex_h.target = vertex_i
    
    vertex_i.target = vertex_j
    vertex_j.target = vertex_m
    vertex_k.target = vertex_m
    vertex_l.target = vertex_m
    vertex_m.target = vertex_n
    '''
    vertices = [
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
        vertex_l,
        vertex_m,
        #vertex_n,
    ]
    
    return vertices

def monster_room_a_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomA-v0')
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

def monster_hall_a_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterHallA-v0')
    env = Waypointer(env, vertices=monster_hall_a_vertices())
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

def monster_hall_b_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterHallB-v0')
    env = Waypointer(env, vertices=monster_hall_b_vertices())
    if process:
        env = ProcessFrame(env, 84, 84)
    env.max_steps = 72
    return env

#def monster_hall_b_expert(process=False):
#    env = gym.make('ImpossiblyGoodVizDoomMonsterHallB-v0')
#    vertices = monster_hall_a_vertices()
#    for v in vertices:
#        v.p[0] = v.p[0] * -1
#    env = Waypointer(env, vertices=vertices)
#    if process:
#        env = ProcessFrame(env, 84, 84)
#    env.max_steps = 72
#    return env

def monster_room_a_easy_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomAEasy-v0')
    env = Waypointer(env, vertices=monster_room_a_vertices())
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

def monster_room_small_a_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomSmallA-v0')
    vertices = monster_room_small_a_vertices()
    env = Waypointer(env, vertices=vertices)
    env.max_steps=32
    return env

def monster_room_small_b_expert(process=False):
    env = gym.make('ImpossiblyGoodVizDoomMonsterRoomSmallB-v0')
    vertices = monster_room_small_a_vertices()
    for v in vertices:
        v.p[0] = v.p[0] * -1
    env = Waypointer(env, vertices=vertices)
    env.max_steps=32
    return env

def monster_room():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterRoomAExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterRoomBExpert-v0')
    return EnvSwitcher(enva, envb)

def monster_room_easy():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterRoomAEasyExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterRoomBEasyExpert-v0')
    return EnvSwitcher(enva, envb)

def monster_room_small():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterRoomSmallAExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterRoomSmallBExpert-v0')
    return EnvSwitcher(enva, envb)

def monster_hall():
    enva = gym.make('ImpossiblyGoodVizDoomMonsterHallAExpert-v0')
    envb = gym.make('ImpossiblyGoodVizDoomMonsterHallBExpert-v0')
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
        id='ImpossiblyGoodVizDoomMonsterRoomSmallA-v0',
        #entry_point='envs.vizdoom:monster_room_a_expert',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_small_a.cfg', 'frame_skip':8},
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
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomSmallAExpert-v0',
        entry_point='envs.vizdoom:monster_room_small_a_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
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
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomSmallBExpert-v0',
        entry_point='envs.vizdoom:monster_room_small_b_expert',
        #entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        #kwargs={'scenario_file':'monster_room_a.cfg', 'frame_skip':8},
    )
    
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
        id='ImpossiblyGoodVizDoomMonsterRoomSmallB-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_room_small_b.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoom-v0',
        entry_point='envs.vizdoom:monster_room',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterHall-v0',
        entry_point='envs.vizdoom:monster_hall',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterHallAExpert-v0',
        entry_point='envs.vizdoom:monster_hall_a_expert',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterHallBExpert-v0',
        entry_point='envs.vizdoom:monster_hall_b_expert',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterHallA-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_hall_a.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterHallB-v0',
        entry_point='vizdoom.gym_wrapper.gym_env_defns:VizdoomScenarioEnv',
        kwargs={'scenario_file':'monster_hall_b.cfg', 'frame_skip':8},
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomEasy-v0',
        entry_point='envs.vizdoom:monster_room_easy',
    )
    
    register(
        id='ImpossiblyGoodVizDoomMonsterRoomSmall-v0',
        entry_point='envs.vizdoom:monster_room_small',
    )
