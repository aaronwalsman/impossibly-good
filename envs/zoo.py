from importlib.metadata import entry_points
from xml.etree.ElementTree import TreeBuilder
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
from gym_minigrid.envs import DoorKeyEnv

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
    register(
        id='ImpossiblyGood-ExampleTwo-7x7-v0',
        entry_point='envs.zoo:ExampleTwo7x7',
    )
    register(
        id='ImpossiblyGood-ExampleTwoEasy-7x7-v0',
        entry_point='envs.zoo:ExampleTwoEasy7x7',
    )
    register(
        id='ImpossiblyGood-ExampleTwoMedium-7x7-v0',
        entry_point='envs.zoo:ExampleTwoMedium7x7',
    )
    register(
        id='ImpossiblyGood-ExampleTwo-9x9-v0',
        entry_point='envs.zoo:ExampleTwo9x9',
    )
    register(
        id='ImpossiblyGood-ExampleThree-7x7-v0',
        entry_point='envs.zoo:ExampleThree7x7',
    )
    register(
        id='ImpossiblyGood-ExampleThreeEasy-9x9-v0',
        entry_point='envs.zoo:ExampleThreeEasy9x9',
    )
    register(
        id='ImpossiblyGood-ExampleThreeMed-9x9-v0',
        entry_point='envs.zoo:ExampleThreeMed9x9',
    )
    register(
        id='ImpossiblyGood-ExampleThreeHard-9x9-v0',
        entry_point='envs.zoo:ExampleThreeHard9x9',
    )
    register(
        id='ImpossiblyGood-ExampleFour-9x9-v0',
        entry_point='envs.zoo:ExampleFour9x9',
    )
    register(
        id='ImpossiblyGood-ExampleFive-9x9-v0',
        entry_point='envs.zoo:ExampleFive9x9',
    )
    register(
        id='ImpossiblyGood-ExampleSix-9x9-v0',
        entry_point='envs.zoo:ExampleSix9x9',
    )
    register(
        id='ImpossiblyGood-ExampleSixEasy-9x9-v0',
        entry_point='envs.zoo:ExampleSixEasy9x9',
    )
    register(
        id='ImpossiblyGood-ExampleSeven-9x9-v0',
        entry_point='envs.zoo:ExampleSeven9x9',
    )
    register(
        id='ImpossiblyGood-ExampleEight-9x9-v0',
        entry_point='envs.zoo:ExampleEight9x9',
    )
    register(
        id='ImpossiblyGood-ExampleEightHard-9x9-v0',
        entry_point='envs.zoo:ExampleEightHard9x9',
    )
    register(
        id='ImpossiblyGood-ExampleNine-9x9-v0',
        entry_point='envs.zoo:ExampleNine9x9',
    )
    register(
        id='ImpossiblyGood-DoorKeyExpert-5x5-v0',
        entry_point='envs.zoo:DoorKeyExpertEnv5x5',
    )

class MatchingColorEnv(MiniGridEnv):
    def __init__(self,
        unseen_color='grey',
        random_colors=('red','blue'),
        #switching_horizon=10,
        **kwargs,
    ):
        mission_space = MissionSpace(
            mission_func=lambda : (
                'move next to the door that has the same color as the balls')
        )
        
        # set the max steps based on the grid size
        if 'grid_size' in kwargs:
            width = kwargs['grid_size']
            height = kwargs['grid_size']
        elif 'width' in kwargs and 'height' in kwargs:
            width = kwargs['width']
            height = kwargs['height']
        else:
            raise ValueError(
                'Requires either "grid_size" or "height" and "width"')
        kwargs.setdefault('max_steps', width*height*2)
        
        # set the goal color randomly
        assert unseen_color not in random_colors
        self.unseen_color = unseen_color
        self.random_colors = random_colors
        #self.switching_horizon = switching_horizon or kwargs['max_steps']
        
        # initialize normally
        super().__init__(
            mission_space=mission_space,
            **kwargs
        )
        
        # modify the obsevation space with the extra 'observed_color' variable
        self.observation_space['observed_color'] = Discrete(len(COLOR_TO_IDX))
        self.observation_space['expert'] = self.action_space
        self.observation_space['step'] = Discrete(self.max_steps)
        #self.observation_space['switching_time'] = Discrete(
        #    self.switching_horizon)
        
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
        #self.switching_time = self._rand_int(0, self.switching_horizon)
        
        # update the observation
        self.observed_color = self.unseen_color
        # breakpoint()
        obs['observed_color'] = self.update_observed_color(obs)
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        #obs['switching_time'] = self.switching_time
        
        return obs
    
    def step(self, action):
        obs, reward, term, info = super().step(action)
        obs['observed_color'] = self.update_observed_color(obs)
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        #obs['switching_time'] = self.switching_time
        
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
            # breakpoint()
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

class DoorKeyExpertEnv(DoorKeyEnv):
    def __init__(self,
        size=8,
        switching_horizon=None,
        **kwargs
    ):
        # The goal position as specified by the DoorKeyEnv
        self.goal_pos = (size - 2, size - 2)
        
        kwargs.setdefault('max_steps', size**2)
        
        self.switching_horizon = switching_horizon or kwargs['max_steps']
        
        # initialize normally
        super().__init__(
            size=size,
            **kwargs
        )
        
        # add a variable to observation space to indicate whether
        # it's the expert or not
        self.observation_space['observed_color'] = Discrete(2)
        self.observation_space['expert'] = self.action_space
        self.observation_space['step'] = Discrete(self.max_steps)
        self.observation_space['switching_time'] = Discrete(
            self.switching_horizon)
    
    def reset(self, *, seed=None, options=None):
        
        self.switching_time = self._rand_int(0, self.max_steps)
        
        # reset normally
        # breakpoint()
        obs = super().reset(seed=seed)
        
        # update the expert action
        obs['observed_color'] = 1
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        obs['switching_time'] = self.switching_time
        
        return obs
    
    def step(self, action):
        obs, reward, term, info = super().step(action)
        obs['observed_color'] = 1
        obs['expert'] = self.compute_expert_action()
        obs['step'] = self.step_count
        obs['switching_time'] = self.switching_time
        
        # ax, ay = self.agent_pos
        # tx, ty = self.goal_pos
        
        # if action == self.actions.toggle:
        #     term = True
        
        # vx, vy = self.front_pos
        # if action == self.actions.forward and self.grid.get(vx, vy) and not self.grid.get(vx, vy).can_overlap():
        #     term = True
        #     reward = 0
        
        # if action == self.actions.done:
        #     term = True
        #     if (ax == tx and abs(ay-ty) == 1) or (ay == ty and abs(ax-tx) == 1):
        #         reward = self._reward()
        
        # print(term)
        # print(reward)
        # breakpoint()
        
        return obs, reward, term, info
    
    def compute_expert_action(self):
        tx, ty = self.goal_pos
        goal_positions = (tx, ty)
        
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
            if (x,y) == goal_positions:
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
            if (forward_cell is None or
                forward_cell.type == 'door' or
                forward_cell.type == 'key' or
                forward_cell.can_overlap()
            ):
                forward_pose = (fx, fy, direction)
                if forward_pose not in closed_set:
                    forward_path = path[:]
                    forward_path.append(self.actions.forward)
                    open_set.append((forward_pose, forward_path))
        
        return path[0]

class DoorKeyExpertEnv5x5(DoorKeyExpertEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, size=5, **kwargs)

def ColoredWall(color):
    class CWall(Wall):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, color=color)
    return CWall

class ExampleOne5x5(MatchingColorEnv):
    '''
    XXXXX
    XXROX
    X^  X
    XXOBX
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
        self.put_obj(Ball(color=self.goal_color), 3, 1)
        self.put_obj(Ball(color=self.goal_color), 2, 3)
        
        # doors
        self.put_obj(Door(color='blue'), 3, 3)
        self.put_obj(Door(color='red'), 2, 1)
        
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
        self.grid.horz_wall(1, 2, length=3, obj_type=ColoredWall('green'))
        self.grid.horz_wall(1, 4, length=3, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 1, length=2)
        self.grid.horz_wall(3, 5, length=2)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 4, 2)
        self.put_obj(Ball(color=self.goal_color), 4, 4)
        
        # doors
        self.put_obj(Door(color='blue'), 5, 5)
        self.put_obj(Door(color='red'), 5, 1)
        
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
        self.grid.horz_wall(1, 3, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(1, 5, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(5, 2, length=2)
        self.grid.horz_wall(5, 6, length=2)
        self.grid.horz_wall(6, 1, length=1)
        self.grid.horz_wall(6, 7, length=1)
        
        # balls
        self.put_obj(Ball(color=self.goal_color), 6, 3)
        self.put_obj(Ball(color=self.goal_color), 6, 5)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 7)
        self.put_obj(Door(color='red'), 7, 1)
        
        # agent
        self.place_agent(top=(1,4), size=(1,1))


class ExampleTwoEasy7x7(MatchingColorEnv):
    '''
    XXXXXXX
    X  O  X
    X RX  X
    X^XXXXX
    X BX  X
    X  O  X
    XXXXXXX
    PPO can do this fine
    n_distill_plus_r got up to 80-something
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
        self.grid.horz_wall(2, 3, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(3, 2, length=1, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 4, length=1, obj_type=ColoredWall('purple'))

        # balls
        self.put_obj(Ball(color=self.goal_color), 3, 1)
        self.put_obj(Ball(color=self.goal_color), 3, 5)

        # doors
        self.put_obj(Door(color='blue'), 2, 2)
        self.put_obj(Door(color='red'), 2, 4)

        # agent
        self.place_agent(top=(1,3), size=(1,1))


class ExampleTwoMedium7x7(MatchingColorEnv):
    '''
    XXXXXXX
    X   O X
    X RX  X
    X^XXXXX
    X BX  X
    X   O X
    XXXXXXX
    PPO cannot handle this at all
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
        self.grid.horz_wall(2, 3, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(3, 2, length=1, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 4, length=1, obj_type=ColoredWall('purple'))

        # balls
        self.put_obj(Ball(color=self.goal_color), 4, 1)
        self.put_obj(Ball(color=self.goal_color), 4, 5)

        # doors
        self.put_obj(Door(color='blue'), 2, 2)
        self.put_obj(Door(color='red'), 2, 4)

        # agent
        self.place_agent(top=(1,3), size=(1,1))


class ExampleTwo7x7(MatchingColorEnv):
    '''
    XXXXXXX
    X    RX
    X XXOXX
    X^XXXXX
    X XXOXX
    X    BX
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
        self.grid.horz_wall(2, 3, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 2, length=2, obj_type=ColoredWall('green'))
        self.grid.horz_wall(5, 2, length=1, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 4, length=2, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(5, 4, length=1, obj_type=ColoredWall('purple'))

        # balls
        self.put_obj(Ball(color=self.goal_color), 4, 2)
        self.put_obj(Ball(color=self.goal_color), 4, 4)

        # doors
        self.put_obj(Door(color='blue'), 5, 5)
        self.put_obj(Door(color='red'), 5, 1)

        # agent
        self.place_agent(top=(1,3), size=(1,1))
    

class ExampleTwo9x9(MatchingColorEnv):
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
        self.grid.vert_wall(2, 2, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 2, length=2, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(6, 2, length=2, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(4, 3, length=3, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(4, 5, length=3, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(3, 6, length=2, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(6, 6, length=2, obj_type=ColoredWall('yellow'))

        # balls
        self.put_obj(Ball(color=self.goal_color), 5, 2)
        self.put_obj(Ball(color=self.goal_color), 5, 6)

        # doors
        self.put_obj(Door(color='blue'), 7, 7)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(1,4), size=(1,1))

class ExampleThreeOld7x7(MatchingColorEnv):
    '''
    XXXXXXX
    XR    X
    X XXXXX
    X^ B  X
    X XXXXX
    X  O  X
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
        self.grid.horz_wall(2, 2, length=4, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 4, length=4, obj_type=ColoredWall('purple'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 3, 5)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 1)
        self.put_obj(Door(color='red'), 3, 3)

        # agent
        self.place_agent(top=(1,3), size=(1,1))

class ExampleThree7x7(MatchingColorEnv):
    '''
    XXXXXXX
    X B   X
    X XXXXX
    X^  R X
    X XXXXX
    X  O  X
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
        self.grid.horz_wall(2, 2, length=4, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 4, length=4, obj_type=ColoredWall('purple'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 3, 5)
        
        # doors
        self.put_obj(Door(color='blue'), 2, 1)
        self.put_obj(Door(color='red'), 4, 3)

        # agent
        self.place_agent(top=(1,3), size=(1,1))

class ExampleThreeEasy9x9(MatchingColorEnv):
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
        self.grid.horz_wall(2, 2, length=6, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 3, length=6, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(2, 5, length=6, obj_type=ColoredWall('yellow'))
        self.grid.vert_wall(2, 6, length=2, obj_type=ColoredWall('blue'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 7)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 4)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(1,4), size=(1,1))

class ExampleThreeMed9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    X      RX
    X XXXXXXX
    X XXXXXXX
    X^     BX
    X XXXXXXX
    X XXXXXXX
    X  O    X
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
        self.grid.horz_wall(2, 2, length=6, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 3, length=6, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(2, 5, length=6, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 6, length=6, obj_type=ColoredWall('blue'))
        #self.grid.vert_wall(2, 6, length=2)
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 3, 7)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 4)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(1,4), size=(1,1))

class ExampleThreeHard9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    X      RX
    X XXXXXXX
    X XXXXXXX
    X^     BX
    X XXXXXXX
    X XXXXXXX
    X      OX
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
        self.grid.horz_wall(2, 2, length=6, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 3, length=6, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(2, 5, length=6, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 6, length=6, obj_type=ColoredWall('blue'))
        #self.grid.vert_wall(2, 6, length=2)
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 7, 7)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 4)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(1,4), size=(1,1))

class ExampleFour9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    XXXXXO XX
    XXXXXX XX
    X      RX
    X^XXXXXXX
    X      BX
    XXXXXX XX
    XXXXXO XX
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
        self.grid.horz_wall(1, 1, length=4, obj_type=ColoredWall('green'))
        self.grid.horz_wall(1, 2, length=5, obj_type=ColoredWall('green'))
        
        self.grid.horz_wall(2, 4, length=6, obj_type=ColoredWall('yellow'))
        
        self.grid.horz_wall(1, 7, length=4, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(1, 6, length=5, obj_type=ColoredWall('purple'))
        
        self.grid.vert_wall(7, 1, length=2, obj_type=ColoredWall('blue'))
        self.grid.vert_wall(7, 6, length=2, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 5, 1)
        self.put_obj(Ball(color=self.goal_color), 5, 7)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 3)
        self.put_obj(Door(color='red'), 7, 5)

        # agent
        self.place_agent(top=(1,4), size=(1,1))

class ExampleFive9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    XO     OX
    X   X   X
    X       X
    X X ^ X X
    X       X
    X   X   X
    XB     RX
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
        self.grid.horz_wall(4, 2, length=1, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 4, length=1, obj_type=ColoredWall('yellow'))
        self.grid.vert_wall(6, 4, length=1, obj_type=ColoredWall('purple'))
        self.grid.vert_wall(4, 6, length=1, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        self.put_obj(Ball(color=self.goal_color), 7, 1)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 7)
        self.put_obj(Door(color='red'), 7, 7)

        # agent
        self.place_agent(top=(3,3), size=(3,3))

class ExampleSix9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    XO     RX   
    X XXXX  X   
    X X     X 
    X X ^   X
    X X     X
    X       X
    XB      X
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
        self.grid.horz_wall(2, 2, length=4, obj_type=ColoredWall('yellow'))
        self.grid.vert_wall(2, 3, length=3, obj_type=ColoredWall('green'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 7)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(3,3), size=(3,3))


class ExampleSixEasy9x9(MatchingColorEnv):
    '''
    XXXXXXXXX
    XO     RX   
    X XXXX  X   
    X X     X 
    X X ^XXXX  
    X X  XXXX   
    X    XXXX
    XB   XXXX
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
        self.grid.horz_wall(2, 2, length=4, obj_type=ColoredWall('yellow'))
        self.grid.vert_wall(2, 3, length=3, obj_type=ColoredWall('green'))
        self.grid.vert_wall(5, 4, length=4, obj_type=ColoredWall('purple'))
        self.grid.vert_wall(6, 4, length=4, obj_type=ColoredWall('red'))
        self.grid.vert_wall(7, 4, length=4, obj_type=ColoredWall('purple'))
        self.grid.vert_wall(8, 4, length=4, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 7)
        self.put_obj(Door(color='red'), 7, 1)

        # agent
        self.place_agent(top=(3,3), size=(3,3))

class ExampleSeven9x9(MatchingColorEnv):        
    '''
    XXXXXXXXX
    XO R    X
    X    X  X 
    XB XXX  X 
    X  XX^  X 
    X X     X
    X X  X  X
    X       X
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
        self.grid.horz_wall(3, 4, length=2, obj_type=ColoredWall('yellow'))
        self.grid.vert_wall(5, 2, length=2, obj_type=ColoredWall('green'))
        self.grid.horz_wall(3, 3, length=2, obj_type=ColoredWall('purple'))
        self.grid.vert_wall(2, 5, length=2, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(5, 6, length=1, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        
        # doors
        self.put_obj(Door(color='blue'), 1, 3)
        self.put_obj(Door(color='red'), 3, 1)

        # agent
        self.place_agent(top=(5,5), size=(3,3))


class ExampleEight9x9(MatchingColorEnv):        
    '''
    XXXXXXXXX
    XO      X
    X  XXXX X  
    X XXXXX X 
    X^    R X 
    X X   B X
    X XXXXX X
    X       X
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
        self.grid.horz_wall(3, 2, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 3, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 5, length=1, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(2, 6, length=5, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        
        # doors
        self.put_obj(Door(color='red'), 6, 4)
        self.put_obj(Door(color='blue'), 6, 5)

        # agent
        self.place_agent(top=(1,4), size=(1,1))


class ExampleEightHard9x9(MatchingColorEnv):        
    '''
    XXXXXXXXX
    XO      X
    X  XXXX X  
    X XXXXX X 
    X^    R X 
    X X   B X
    X  XXXX X
    XO      X
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
        self.grid.horz_wall(3, 2, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 3, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 5, length=1, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(3, 6, length=5, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 1)
        self.put_obj(Ball(color=self.goal_color), 1, 7)
        
        # doors
        self.put_obj(Door(color='red'), 6, 4)
        self.put_obj(Door(color='blue'), 6, 5)

        # agent
        self.place_agent(top=(1,4), size=(1,1))


class ExampleNine9x9(MatchingColorEnv):        
    '''
    XXXXXXXXX
    XR      X
    X  XXXO X  
    X XXXXX X 
    X^      X 
    X XXXXX X
    X  XXXO X
    XB      X
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
        self.grid.horz_wall(3, 2, length=4, obj_type=ColoredWall('yellow'))
        self.grid.horz_wall(2, 3, length=5, obj_type=ColoredWall('green'))
        self.grid.horz_wall(2, 5, length=5, obj_type=ColoredWall('purple'))
        self.grid.horz_wall(3, 6, length=3, obj_type=ColoredWall('red'))
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 6, 6)
        # self.put_obj(Ball(color=self.goal_color), 1, 7)
        
        # doors
        self.put_obj(Door(color='red'), 1, 1)
        self.put_obj(Door(color='blue'), 1, 7)

        # agent
        self.place_agent(top=(1,4), size=(1,1))


class TestIgnoreDoor(DoorKeyExpertEnv):
    '''
    XXXXXXXXX
    X       X
    X XXXXXXX
    X X     X 
    X^   R BX
    X XXXXXXX
    X X     X
    XOX     X
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
        self.grid.vert_wall(2, 2, length=2)
        self.grid.horz_wall(2, 2, length=5)
        self.grid.horz_wall(2, 5, length=6)
        self.grid.vert_wall(2, 6, length=2)
        
        # balls 
        self.put_obj(Ball(color=self.goal_color), 1, 7)
        
        # doors
        self.put_obj(Door(color='blue'), 7, 4)
        self.put_obj(Door(color='red'), 5, 4)

        # agent
        self.place_agent(top=(1,4), size=(1,1))
