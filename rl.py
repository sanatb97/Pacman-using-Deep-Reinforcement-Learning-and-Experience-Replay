import numpy as np
# import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import gym
from keras.models import Sequential
from keras.layers.core import Dense, Activation
#from keras.optimizers import RMSprop
import random
import os 
import pickle
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
env = gym.make('MsPacman-ram-v0')

model = Sequential()
model.add(Dense(128, init='lecun_uniform', input_shape=(128,)))
model.add(Activation('relu'))

model.add(Dense(9, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

model.compile(loss='mse', optimizer='Adam')

num_episodes = 1000
gamma = 0.9
epsilon = 1
epsilon_decay = 0.999
rewards = []
epsilons = []
for i in range(num_episodes):
   
    state = env.reset()
    done = False
    tot_reward_per_eposide = 0
    while not done:
        env.render()
        qval = model.predict(state.reshape(1,128), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else:
            action = (np.argmax(qval))
        new_state, reward, done, _ = env.step(action)
        #Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1,128), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1,9))
        y[:] = qval[:]
        if not done:
            update = (reward + (gamma * maxQ))
        else:
            update = reward
        y[0][action] = update
        model.fit(state.reshape(1,128), y, batch_size=1, verbose=0)
        state = new_state
        tot_reward_per_eposide += reward
    rewards.append(tot_reward_per_eposide)

    #print(tot_reward_per_eposide)
    if epsilon > 0.1:
        epsilon *= epsilon_decay
    epsilons.append(epsilon)
    print("episode: ",i," score: ", tot_reward_per_eposide, " epsilon: ",epsilon)

with open("rewards_rl.pkl",'wb') as f:
    pickle.dump(rewards, f)

with open("epsilons_rl.pkl",'wb') as f1:
    pickle.dump(epsilons, f1)


