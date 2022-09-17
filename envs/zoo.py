import numpy

from gym.envs.registration import register
from gym.spaces import Dict, Discrete
from gym_minigrid.minigrid import (
    Wall,
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

from PIL import Image

class ExpertException(Exception):
    pass

def register_impossibly_good_envs():
    register(
        id='ImpossiblyGood-ExampleOne-5x5-v0',
        entry_point='envs.zoo:ExampleOne5x5',
    )
    register(
        id='ImpossiblyGood-ExampleOne-7x7-v0',
        entry_point='envs.zoo:ExampleOne7x7',
    )
    register(
        id='ImpossiblyGood-ExampleOne-9x9-v0',
        entry_point='envs.zoo:ExampleOne9x9',
    )

class MatchingColorEnv(MiniGridEnv):
    def __init__(self,
        unseen_color='grey',
        random_colors=('red','blue'),
        switching_horizon=10,
        **kwargs,
    ):
        mission_space = MissionSpace(
            mission_func=lambda : (
                'move next to the door that has the same color as the balls')
        )
        
        # set the goal color randomly
        assert unseen_color not in random_colors
        self.unseen_color = unseen_color
        self.random_colors = random_colors
        self.switching_horizon = switching_horizon
        
        kwargs.setdefault('max_steps', 200)
        
        # initialize normally
        super().__init__(
            mission_space=mission_space,
            **kwargs
        )
        
        # modify the obsevation space with the extra 'observed_color' variable
        self.observation_space['observed_color'] = Discrete(len(COLOR_TO_IDX))
        self.observation_space['expert'] = self.action_space
        self.observation_space['step'] = Discrete(self.max_steps)
        self.observation_space['switching_time'] = Discrete(
            self.switching_horizon)
        
        # mission
        self.mission = 'enter the door that has the same color as the balls'
    
    def reset(self, *, seed=None, options=None):
        # pick a goal color for this episode
        self.goal_color = self.random_colors[
            self._rand_int(0,len(self.random_colors))]
        
        # reset normally
        obs = super().reset(seed=seed)
        
        # update the goal position
        self.update_goal_pos()
        
        # update the switching time
        # this is used for training the follower/explorer model and is
        # more useful to keep track of here than elsewhere in the code
        self.switching_time = self._rand_int(0, self.max_steps)
        
        # update the observation
        self.observed_color = self.unseen_color
        obs['observed_color'] = self.update_observed_color(obs)
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        obs['switching_time'] = self.switching_time
        
        return obs
    
    def step(self, action):
        obs, reward, term, info = super().step(action)
        obs['observed_color'] = self.update_observed_color(obs)
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        obs['switching_time'] = self.switching_time
        
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
            observed_color_idx = obs['image'][:,:,1][ball_x[0], ball_y[0]]
            self.observed_color = IDX_TO_COLOR[observed_color_idx]
        
        return COLOR_TO_IDX[self.observed_color]
    
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
            except IndexError:
                #image = Image.fromarray(self.render('rgb_array'))
                #image.save('./no_path.png')
                #raise ExpertException(
                #    'No path to goal. This should never happen.')
                # this actually can happen if the agent moves a ball in the way
                # of the goal
                path = [self.actions.done]
                break
                
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

class ExampleOne5x5(MatchingColorEnv):
    '''
    XXXXX
    XXBOX
    X^  X
    XXORX
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
        
        # extra walls
        self.grid.horz_wall(1, 1, length=1)
        self.grid.vert_wall(1, 3, length=1)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 2, 1)
        self.put_obj(Ball(color=self.goal_color), 3, 3)
        
        # doors
        self.put_obj(Door(color='blue'), 2, 3)
        self.put_obj(Door(color='red'), 3, 1)
        
        # agent
        self.place_agent(top=(1,2), size=(1,1))

class ExampleOne7x7(MatchingColorEnv):
    '''
    XXXXXXX
    X  XXRX
    XXXXO X
    X^    X
    XXXXO X
    X  XXBX
    XXXXXXX
    '''
    def __init__(self, **kwargs):
        kwargs['grid_size'] = 7
        kwargs['agent_view_size'] = 3
        super().__init__(**kwargs)
    
    def _gen_grid(self, width, height):
        
        # initialize grid
        self.grid = Grid(width, height)
        
        # build surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # walls
        def ColoredWall(color):
            class CWall(Wall):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs, color=color)
            return CWall
        
        self.grid.horz_wall(1, 2, length=3, obj_type=ColoredWall('green'))
        self.grid.horz_wall(1, 4, length=3, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 1, length=2)
        self.grid.horz_wall(3, 5, length=2)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 4, 2)
        self.put_obj(Ball(color=self.goal_color), 4, 4)
        
        # doors
        self.put_obj(Door(color='blue'), 5, 1)
        self.put_obj(Door(color='red'), 5, 5)
        
        # agent
        self.place_agent(top=(1,3), size=(1,1))
    

class ExampleOne9x9(MatchingColorEnv):
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
        self.put_obj(Door(color='blue'), 7, 1)
        self.put_obj(Door(color='red'), 7, 7)
        
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
