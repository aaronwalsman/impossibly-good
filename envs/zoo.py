import numpy

from gym.envs.registration import register
from gym.spaces import Dict, Discrete
from gym_minigrid.minigrid import (
    Ball,
    Door,
    Grid,
    MiniGridEnv,
    MissionSpace,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
    DIR_TO_VEC,
)

def register_impossibly_good_envs():
    register(
        id='ImpossiblyGood-ExampleOne-v0',
        entropy_point='impossibly_good.envs.zoo:ExampleOne',
    )

class MatchingColorEnv(MiniGridEnv):
    def __init__(self,
        unseen_color='grey',
        random_colors=('red','blue'),
        **kwargs
    ):
        mission_space = MissionSpace(
            mission_func=lambda : (
                'move next to the door that has the same color as the balls')
        )
        
        # set the goal color randomly
        assert unseen_color not in random_colors
        self.unseen_color = unseen_color
        self.random_colors = random_colors
        
        # initialize normally
        super().__init__(
            mission_space=mission_space,
            **kwargs
        )
        
        # modify the obsevation space with the extra 'observed_color' variable
        self.observation_space['observed_color'] = Discrete(len(COLOR_TO_IDX))
        self.observation_space['expert'] = self.action_space
        
        # mission
        self.mission = 'enter the door that has the same color as the balls'
    
    def reset(self):
        # pick a goal color for this episode
        self.goal_color = self.random_colors[
            self._rand_int(0,len(self.random_colors))]
        
        # reset normally
        obs = super().reset()
        
        # update the goal position
        self.update_goal_pos()
        
        # update the observed color
        self.observed_color = self.unseen_color
        obs['observed_color'] = self.update_observed_color(obs)
        
        # update the expert action
        obs['expert'] = self.compute_expert_action()
        
        return obs
    
    def step(self, action):
        obs, reward, term, info = super().step(action)
        obs['observed_color'] = self.update_observed_color(obs)
        obs['expert'] = self.compute_expert_action()
        
        ax, ay = self.agent_pos
        tx, ty = self.goal_pos
        
        if action == self.actions.toggle:
            term = True
        
        if action == self.actions.done:
            term = True
            if (ax == tx and abs(ay-ty) == 1) or (ay == ty and abs(ax-tx) == 1):
                reward = self._reward()
        
        return obs, reward, term, info
    
    def update_goal_pos(self):
        full_grid = self.grid.encode()
        goal_x, goal_y = numpy.where(
            (full_grid[:,:,0] == OBJECT_TO_IDX['door']) &
            (full_grid[:,:,1] == COLOR_TO_IDX[self.goal_color])
        )
        assert len(goal_x) == 1
        self.goal_pos = (int(goal_x[0]), int(goal_y[0]))
    
    def update_observed_color(self, obs):
        # check if the colored ball is visible
        ball_x, ball_y = numpy.where(
            obs['image'][:,:,0] == OBJECT_TO_IDX['ball'])
        if len(ball_x):
            self.observed_color = obs['image'][:,:,1][ball_x[0], ball_y[0]]
        
        return self.observed_color
    
    def compute_expert_action(self):
        tx, ty = self.goal_pos
        goal_positions = set(((tx+1, ty), (tx, ty+1), (tx-1, ty), (tx, ty-1)))
        
        # compute shortest path
        agent_pose = self.agent_pos[0], self.agent_pos[1], self.agent_dir
        open_set = [(agent_pose, [])]
        closed_set = set()
        while True:
            try:
                pose, path = open_set.pop(0)
            except:
                import pdb
                pdb.set_trace()
            closed_set.add(pose)
            x, y, direction = pose
            if (x,y) in goal_positions:
                path.append(self.actions.done)
                break
            
            # left
            left_pose = (x, y, (direction-1)%4)
            if left_pose not in closed_set:
                left_path = path[:]
                left_path.append(self.actions.left)
                open_set.append((left_pose, left_path))
            
            # right
            right_pose = (x, y, (direction+1)%4)
            if right_pose not in closed_set:
                right_path = path[:]
                right_path.append(self.actions.right)
                open_set.append((right_pose, right_path))
            
            # forward
            vx, vy = DIR_TO_VEC[direction]
            fx = x + vx
            fy = y + vy
            forward_cell = self.grid.get(fx, fy)
            if forward_cell is None or forward_cell.can_overlap():
                forward_pose = (fx, fy, direction)
                if forward_pose not in closed_set:
                    forward_path = path[:]
                    forward_path.append(self.actions.forward)
                    open_set.append((forward_pose, forward_path))
        
        return path[0]

class ExampleZero(MatchingColorEnv):
    '''
    XXXXX
    XB  X
    X  RX
    XO^ X
    XXXXX
    '''
    def __init__(self, **kwargs):
        kwargs['grid_size'] = 5
        kwargs['agent_view_size'] = 3
        super().__init__(**kwargs)
    
    def _gen_grid(self, width, height):
        
        # initialize_grid
        self.grid = Grid(width, height)
        
        # build surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 3)
        self.put_obj(Door(color='red'), 3, 2)
        
        # agent
        self.place_agent(top=(2,1), size=(1,1))

class ExampleOne(MatchingColorEnv):
    '''
    XXXXXXXXX
    X     XRX
    X    XX X
    XXXXXXO X
    X^      X
    XXXXXXO X
    X    XX X
    X     XBX
    XXXXXXXXX
    '''
    def __init__(self, **kwargs):
        kwargs['grid_size'] = 9
        kwargs['agent_view_size'] = 3
        super().__init__(**kwargs)
    
    def _gen_grid(self, width, height):
        
        # initialize grid
        self.grid = Grid(width, height)
        
        # build surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # walls
        self.grid.horz_wall(1, 3, length=5)
        self.grid.horz_wall(1, 5, length=5)
        self.grid.horz_wall(5, 2, length=2)
        self.grid.horz_wall(5, 6, length=2)
        self.grid.horz_wall(6, 1, length=1)
        self.grid.horz_wall(6, 7, length=1)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 6, 3)
        self.put_obj(Ball(color=self.goal_color), 6, 5)
        
        # doors
        self.put_obj(Goal(color='blue'), 7, 1)
        self.put_obj(Goal(color='red'), 7, 7)
        
        # agent
        self.place_agent(top=(1,4), size=(1,1))

class example_two(MiniGridEnv):
    '''
    XXXXXXXXX
    X      RX
    X XXXOXXX
    X X XXX X
    X^X     X
    X X XXX X
    X XXXOXXX
    X      BX
    XXXXXXXXX
    '''
    
    def _gen_grid(self, width, height):
        # initialize grid
        self.grid = Grid(width, height)
        
        # build surrounding walls
        self.grid.wall_rect(0,0,width,height)
        
        # walls
        self.grid.vert_wall(2, 2, length=5)
        self.grid.horz_wall(3, 2, length=2)

class example_three(MiniGridEnv):
    '''
    XXXXXXXXX
    X      RX
    X XXXXXXX
    X XXXXXXX
    X^     BX
    X XXXXXXX
    X X     X
    XOX     X
    XXXXXXXXX
    '''
