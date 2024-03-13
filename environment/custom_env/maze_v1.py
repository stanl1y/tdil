import gym
from gym import spaces
import numpy as np

class Maze_v1(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, maze_size=100, random_reset=True):
        super(Maze_v1, self).__init__()
        
        self.maze_size = maze_size
        self.random_reset=random_reset
        # left, up, right, down
        self.ACTIONS = [np.array([-1, 0]),
                        np.array([0 , 1]),
                        np.array([1 , 0]),
                        np.array([0 , -1])]

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=maze_size, shape=(4,), dtype=np.uint8)

    def is_terminal(self):
        return self.current_goal==4

    def expert_step(self):
        target=self.key_list[self.current_goal]
        if self.current_goal==1 and self.state[1]<51:#still in the original room
            target=np.array([24 if target[0]<=24 else 25, 51])
        elif self.current_goal==2 and self.state[0]<51:
            target=np.array([51, 74 if target[1]<=74 else 75])
        elif self.current_goal==3 and self.state[1]>48:
            target=np.array([74 if target[0]<=74 else 75, 48])

        dist=target-self.state
        if dist[0]>0:
            ver_action=2
        else:
            ver_action=0

        if dist[1]>0:
            hor_action=1
        else:
            hor_action=3
        
        if abs(dist[0])>abs(dist[1]):
            if self.check_legal(self.state+self.ACTIONS[ver_action]):
                action=ver_action
            else:
                action=hor_action
        else:
            if self.check_legal(self.state+self.ACTIONS[hor_action]):
                action=hor_action
            else:
                action=ver_action

        return action


    def check_legal(self,state):
        x, y = state
        if 49<=x and x<=50 and not(74<=y and y<=75 and self.current_goal==2):#check horizontal wall
            return False
        if 49<=y and y<=50 and not(24<=x and x<=25 and self.current_goal==1) and not(74<=x and x<=75 and self.current_goal==3):#check vertical wall
            return False
        if x < 0 or x >= self.maze_size or y < 0 or y >= self.maze_size:
            return False
        
        return True
    
    def check_goal(self,state):
        return (state==self.key_list[self.current_goal]).all()

    def step(self, action):
        next_state = (np.array(self.state) + self.ACTIONS[action]).tolist()
        x, y = next_state

        if not self.check_legal([x,y]):#x < 0 or x >= self.maze_size or y < 0 or y >= self.maze_size:
            next_state = self.state
        
        self.state = np.array(next_state)
        reward = 0.0
        if self.check_goal(self.state):
            self.current_goal+=1
            reward = 1.0

        # if self.is_small_reward(self.state):
        #     return self.state, 0.01, True, {}
              
        return np.concatenate((self.state,self.key_list[self.current_goal])), reward, self.is_terminal(), {"x":self.state[0],"y":self.state[1],"goal_idx":self.current_goal,"goal":self.key_list[self.current_goal]}
        
    def reset(self):
        if self.random_reset:
            self.state = np.array([np.random.randint(49),np.random.randint(49)])#(0~48,0~48)
            self.key_list=[np.array([np.random.randint(2,48),np.random.randint(2,48)]),#(2~47,2~47),
                np.array([np.random.randint(2,48),np.random.randint(52,99)]),#(2~47,52~98)
                np.array([np.random.randint(52,99),np.random.randint(52,99)]),#(52~98,52~98)
                np.array([np.random.randint(52,99),np.random.randint(2,48)])#(52~98,2~47)
            ]
            while (self.key_list[0]==self.state).all():
                self.key_list[0]=np.array([np.random.randint(2,48),np.random.randint(2,48)])
        else:
            self.state = np.array([48,0])
            self.key_list=[np.array([20,20]),
                np.array([30,80]),
                np.array([70,65]),
                np.array([80,25])
            ]
        self.current_goal=0
        self.key_list.append(self.key_list[-1])
        return np.concatenate((self.state,self.key_list[self.current_goal]))
