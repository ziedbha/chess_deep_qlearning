import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Reshape,Multiply
from keras.layers.convolutional import Conv2D
import random

from chess_env import *

# Worst case num movements pet board state: 64 movements (from) -> (to) on a 8x8 chess board
num_actions = 64 * 64

# TODO:
# Add movement legality information in input
# # This is partially done
# Remove convolutions, why do we need the network to focus on neighborhood information?

class Q_model():
    def __init__(self):
        self.model = self.create_q_model()

    def create_q_model(self):
        weight_learning_rate = 0.001
        init = tf.keras.initializers.HeUniform(42)

        # 8x8 board with 12 piece types
        state = keras.Input(shape=(8, 8, 12))
        legal_move_mask = keras.Input(shape=(num_actions))
       
        # Convolutions on the frames on the screen 
        x = Conv2D(filters=64,padding="same",kernel_size = 2,strides = (2,2), activation='relu', kernel_initializer=init)(state)
        x = Conv2D(filters=128,padding="same",kernel_size=2,strides = (2,2), activation='relu', kernel_initializer=init)(x)
        x = Conv2D(filters=256,padding="same",kernel_size=2,strides = (2,2), activation='relu', kernel_initializer=init)(x)
        x = Conv2D(filters=512,padding="same",kernel_size=2,strides = (2,2), activation='relu', kernel_initializer=init)(x)
        x = Conv2D(filters=1024,padding="same",kernel_size=2,strides = (2,2), activation='relu', kernel_initializer=init)(x)
        x = Flatten()(x)
        actions_and_q_values = Dense(num_actions, activation = 'linear', kernel_initializer=init)(x)
        actions_and_q_values = Multiply()([actions_and_q_values, legal_move_mask])

        model = keras.Model(inputs=[state, legal_move_mask], outputs=actions_and_q_values)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=weight_learning_rate), metrics=['accuracy'])
        return model

    # Predict q values given input: state of board, legal moves mask
    def forward_pass(self, env):
        legal_moves_mask = tf.convert_to_tensor(env.get_legal_moves_mask())
        legal_moves_mask = tf.expand_dims(legal_moves_mask, 0)
        state_tensor = tf.convert_to_tensor(env.translate_board())
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_and_q_values = self.model([state_tensor, legal_moves_mask], training=False)
        return action_and_q_values[0]
    
    # Predict q values and pick action that leads to highest q value
    def predict_and_pick_best(self, env):
        action_space = self.forward_pass(env)
        action = np.argmax(action_space, axis=None)
        move= num2move[action]
        return move,action
    
    # Randomly pick action, or predict and add noise on predictions (smart prediction)
    # TODO: need the
    def explore(self, env, smart_explore=False):
        action_space = None
        if smart_explore:
            predicted_q_values = self.forward_pass(env)

            # Normalize q values
            norm = np.linalg.norm(predicted_q_values)
            predicted_q_values = predicted_q_values / norm

            # Add noise on normalized q values
            noise_on_q_values = np.random.rand(num_actions)
            noisy_pred = np.add(noise_on_q_values, predicted_q_values)
            action_space = noisy_pred
        else:
            action_space = np.random.rand(num_actions)

        # Filter illegal moves
        legal_moves_mask = env.get_legal_moves_mask()
        action_space = action_space * legal_moves_mask

        # Pick highest q value action
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move,action

    def train(self, replay_memory, model, target_model, done):
        # Replay memory:
        # State, action, reward, new state, done
        learning_rate = 0.90 # Learning rate
        discount_factor = 0.99

        MIN_REPLAY_SIZE = 1000
        if len(replay_memory) < MIN_REPLAY_SIZE:
            return

        batch_size = 64 * 2
        mini_batch = None
        if done:
            mini_batch = [*range(len(replay_memory) - batch_size - 1, len(replay_memory))]
        else:
            mini_batch1 = [*range(len(replay_memory) - int(batch_size/2), len(replay_memory))]
            mini_batch2 = np.random.choice(len(replay_memory), int(batch_size / 2)).tolist()
            mini_batch = mini_batch1 + mini_batch2

        # TODO: tensorflow crash
        current_states = np.array([replay_memory[idx][0] for idx in mini_batch])
        current_qs_list = model.predict(current_states)

        new_current_states = np.array([replay_memory[idx][3] for idx in mini_batch])
        future_qs_list = target_model.predict(new_current_states)
        
        X = []
        Y = []
        # Optimize this loop
        for idx, replay_index in enumerate(mini_batch):
            (state, action, reward, _, done) = replay_memory[replay_index]
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[idx])
            else:
                max_future_q = reward

            current_qs = current_qs_list[idx]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(state)
            Y.append(current_qs)
        return model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)