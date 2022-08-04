from keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
import gym
import numpy as np


class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.action_memory=np.zeros((self.mem_size,n_actions), dtype=np.int8)#OneHotEncoding
        self.state_memory=np.zeros((self.mem_size,input_shape))
        self.next_state_memory=np.zeros((self.mem_size,input_shape))
        self.reward_memory=np.zeros(self.mem_size)
        self.done_memory=np.zeros(self.mem_size)
    
    def store_transition(self,action,state,next_state,reward,done):
        index=self.mem_cntr % self.mem_size
        
        OneHotAction=np.zeros(self.action_memory.shape[1])
        OneHotAction[action]=1
        
        self.action_memory[index]=OneHotAction
        self.state_memory[index]=state
        self.next_state_memory[index]=next_state
        self.reward_memory[index]=reward
        self.done_memory[index]=1-done
        self.mem_cntr += 1
        
    def get_samples(self,batch_size):
        max_mem=min(self.mem_size,self.mem_cntr)
        batch=np.random.choice(max_mem,batch_size)
        actions=self.action_memory[batch]
        states=self.state_memory[batch]
        next_states=self.next_state_memory[batch]
        rewards=self.reward_memory[batch]
        done=self.done_memory[batch]
        return actions,states,next_states,rewards,done
        

 def build_dqn(input_shape,output_shape):
    model = Sequential([
                Dense(256, input_shape=(input_shape,),activation='relu'),
                Dense(256,activation='relu'),
                Dense(output_shape,activation='linear')])

    model.compile(optimizer=Adam(lr=0.0005),loss='mse')
    return model



class Agent(object):
    def __init__(self,mem_size,input_shape,n_actions,epsilon,epsilon_dec,epsilon_min,exploration_rate,batch_size):
        self.memory=ReplayBuffer(mem_size,input_shape,n_actions)
        self.policy_net=build_dqn(input_shape,n_actions)
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_dec=epsilon_dec
        self.exploration_rate=exploration_rate
        self.actions=[i for i in range(n_actions)]
        self.batch_size=batch_size
        
    def choose_action(self,state):
        state=state[np.newaxis,:]
        rand=np.random.random()
        if self.epsilon>rand:
            action=np.random.choice(self.actions)
        else:
            q_vals=self.policy_net.predict(state)
            action=np.argmax(q_vals)
        
        return action
    
    def store_transition(self,action,state,next_state,reward,done):
        self.memory.store_transition(action,state,next_state,reward,done)
    
    def learn(self):
        if self.batch_size < self.memory.mem_cntr :
            actions,states,next_states,rewards,done=self.memory.get_samples(self.batch_size)
            
            q_vals=self.policy_net.predict(states)
            q_next=self.policy_net.predict(next_states)
            q_target=q_vals.copy()
            
            action_values=np.array(self.actions,dtype=np.int8)
            action_indices=np.dot(actions,action_values)
            samples = np.arange(self.batch_size, dtype=np.int32)
            
            q_target[samples,action_indices]=rewards + self.exploration_rate*np.max(q_next, axis=1)*done
                        
            self.policy_net.fit(states,q_target,verbose=0)
            self.epsilon = self.epsilon*self.epsilon_dec
            

#main

env=gym.make('CartPole-v0')
agent=Agent(mem_size=1000000,input_shape=4,n_actions=2,epsilon=1,epsilon_dec=0.995,epsilon_min=0.0,exploration_rate=0.995,batch_size=64)
n_games=30
scores=[]
for i in range(n_games):
    state=env.reset()
    done=False
    score=0
    while not done:
        action=agent.choose_action(state)
        next_state,reward,done,info=env.step(action)
        env.render()
        agent.store_transition(action=action,state=state,next_state=next_state,reward=reward,done=int(done))
        state=next_state
        score+=reward
        agent.learn()
    scores.append(score)
    avg_score = np.mean(scores)
    print('episode: ', i,'score: %.2f' % score,' average score %.2f' % avg_score)
env.close()













