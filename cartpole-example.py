import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

learning_rate = 1e-3
env = gym.make('CartPole-v0')
env.reset()
n_steps = 500
min_score = 50
n_games = 10000


def initial_population():
    training_data = []
    scores = []
    observed_scores = []
    for _ in range(n_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(n_steps):
            action = random.randrange(0, 2)
            observation, reward, done, _ = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        if score >= min_score:
            observed_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)
    return scores, training_data


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    layers = [128, 256, 512, 256, 128]

    for layer in layers:
        network = fully_connected(network, layer, activation='relu')
        network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=False, run_id='openai_learning')
    return model


initial_scores, training_data = initial_population()

model = train_model(training_data)
scores = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(n_steps):
        env.render()

        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])

        new_observation, reward, done, _ = env.step(action)

        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    scores.append(score)
print('Average Initial Score:', sum(initial_scores)/len(initial_scores))
print('Average Score after training:', sum(scores)/len(scores))
