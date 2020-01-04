import gym
import numpy as np
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
from collections import deque
import random
import pickle
num_of_episodes = 1000
discount_factor = 0.99
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.1
batch_size = 128
train_start = 1000
scores = []
epsilons = []
#initialize replay memory
memory = deque(maxlen=2000)


env = gym.make('MsPacman-ram-v0')
state_size = env.observation_space.shape[0]
print("Hey there ", state_size)
num_of_actions = 9 # number of neurons at the o/p layer of the DQN

model = Sequential()
model.add(Dense(128, init='lecun_uniform', input_shape=(128,)))
model.add(Activation('relu'))

model.add(Dense(num_of_actions, init='lecun_uniform'))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='Adam')

def get_action(state):
    """Select an action via exploration or exploitation 
    """
    if random.random() < epsilon:
        return random.randint(0, num_of_actions-1) #exploration
    q_val = model.predict(state.reshape(1, state_size)) #exploitation
    return np.argmax(q_val)

def append_sample(state, action, reward, new_state, done):
    memory.append((state, action, reward, new_state, done))

def train_model():
    if len(memory) < train_start:
        return
    batch = random.sample(memory, batch_size)
    update_input = np.zeros((batch_size, state_size))
    update_target = np.zeros((batch_size, state_size))
    action = [0 for _ in range(batch_size)]
    reward = [0 for _ in range(batch_size)]
    done = [0 for _ in range(batch_size)]
    
    for i in range(batch_size):
        """Store state summary in replay memory"""
        update_input[i] = batch[i][0] 
        action[i] = batch[i][1]
        reward[i] = batch[i][2]
        update_target[i] = batch[i][3]
        done[i] = batch[i][4]

    target = model.predict(update_input)
    target_val = model.predict(update_target)

    for i in range(batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            target[i][action[i]] = reward[i] + discount_factor*(np.amax(target_val[i]))
    model.fit(update_input, target, batch_size=batch_size,epochs=1, verbose=0)
    
if __name__ == "__main__":
    for i in range(num_of_episodes):
        done = False
        state = env.reset()
        sum_episode_reward = 0
        while not done:
            env.render()
            action = get_action(state)
            new_state, reward, done, info = env.step(action)
#            print(reward)
            append_sample(state, action, reward, new_state, done)
            if epsilon > epsilon_min :
                epsilon *= epsilon_decay
            train_model()
            sum_episode_reward += reward
        print("episode: ",i," score: ", sum_episode_reward, " epsilon: ",epsilon)
        scores.append(sum_episode_reward)
        epsilons.append(epsilon)
    with open("rewards_rl_replay.pkl",'wb') as f:
        pickle.dump(scores, f)

    with open("epsilons_rl_replay.pkl",'wb') as f1:
        pickle.dump(epsilons, f1)

