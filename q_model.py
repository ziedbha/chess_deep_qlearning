import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense,Flatten,Add
from keras.layers.convolutional import Conv2D
import random
from collections import deque
from collections import namedtuple
from keras import backend as K

from chess_env import *

# Worst case num movements pet board state: 64 movements (from) -> (to) on a 8x8 chess board
num_actions = 64 * 64 + 16 + 16

# TODO:
# Add movement legality information in input
# # This is partially done
# Remove convolutions, why do we need the network to focus on neighborhood information?

class ChessBrain():
    ChessMemory = namedtuple('ChessMemory', ['current_board', 'current_legal_moves', 'next_board', 'next_legal_moves', 'action', 'reward', 'done', 'is_checkmate'])
    def __init__(self):
        self.replay_memory = deque(maxlen=100_000)
        self.checkmate_index_memory = deque(maxlen=500)
        self.min_replay_memory = 300
    
    def add(self, memory):
        self.replay_memory.append(memory)
        # Record this memory's index in the replay memory if it was a checkmate
        if memory.is_checkmate:
            self.checkmate_index_memory.append(len(self.replay_memory))

    def get(self, index):
        return self.replay_memory[index]
    
    def big_enough(self):
        return len(self.replay_memory) > self.min_replay_memory

    def num_checkmates(self):
        return len(self.checkmate_index_memory)

    def sample_interval(self, start, end):
        return [*range(start, end)]

    def sample_newest(self, n):
        return [*range(len(self.replay_memory) - n, len(self.replay_memory))]

    def sample_random(self, n):
        return np.random.choice(len(self.replay_memory), n).tolist()

    def sample_random_checkmate(self, n):
        checkmate_memory_idx = self.checkmate_index_memory[np.random.choice(len(self.checkmate_index_memory))]

        # If the checkmate memory is *very* old, make sure to increase the sample start index
        if checkmate_memory_idx < n:
            checkmate_memory_idx = n

        # Sample starting at checkmate index going backwards
        return self.sample_interval(checkmate_memory_idx - n, checkmate_memory_idx)

class Q_model():
    def __init__(self):
        self.model = self.create_q_model()
        self.chess_brain = ChessBrain()

    def create_q_model(self):
        weight_learning_rate = 0.9
        init = tf.keras.initializers.HeUniform(42)

        # 8x8 board with 12 piece types
        state = keras.Input(shape=(8, 8, 12), name='input_board')
        legal_move_mask = keras.Input(shape=(num_actions), name='input_legal_moves_mask')
       
        # Convolutions on the frames on the screen 
        x = Conv2D(filters=64,padding="same",kernel_size =3,strides = (2,2), activation='sigmoid', kernel_initializer=init)(state)
        x = Conv2D(filters=128,padding="same",kernel_size=3,strides = (2,2), activation='sigmoid', kernel_initializer=init)(x)
        x = Conv2D(filters=256,padding="same",kernel_size=4,strides = (2,2), activation='sigmoid', kernel_initializer=init)(x)
        x = Flatten()(x)

        # Map q values between 0 --> 1
        actions_and_q_values = Dense(num_actions, activation = 'relu', kernel_initializer=init)(x)

        # Make illegal moves very low in Q value
        actions_and_q_values = Add()([actions_and_q_values, legal_move_mask])
       
        model = keras.Model(inputs=[state, legal_move_mask], outputs=actions_and_q_values)
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=weight_learning_rate, beta_1=0.99), metrics=['accuracy'])
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
        action_space_numpy = action_space.numpy()
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
        action_space = action_space + legal_moves_mask

        # Pick highest q value action
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move,action

    def train(self, model, target_model, done):
        learning_rate = 0.90 # Learning rate
        discount_factor = 0.99

        batch_size = 64 * 2
        if not self.chess_brain.big_enough():
            return
        mini_batch = None

        if done:
            # Sample approx this entire game
            mini_batch = self.chess_brain.sample_newest(batch_size)
        else:
            # Checkmates are few... let's not overtrain
            smart_sampling_rate = 0.3
            if np.random.rand() < smart_sampling_rate and self.chess_brain.num_checkmates() > 2:
                # Sample 2 checkmate memories
                mini_batch1 = self.chess_brain.sample_random_checkmate(int(batch_size/2))
                mini_batch2 = self.chess_brain.sample_random_checkmate(int(batch_size/2))
            else:
                # Sample half batch new, half batch random
                mini_batch1 = self.chess_brain.sample_newest(int(batch_size/2))
                mini_batch2 = self.chess_brain.sample_random(int(batch_size/2))
            mini_batch = mini_batch1 + mini_batch2

        current_boards = np.array([self.chess_brain.get(idx).current_board for idx in mini_batch])
        current_legal_moves = np.array([self.chess_brain.get(idx).current_legal_moves for idx in mini_batch])
        current_qs_list = model.predict([current_boards, current_legal_moves], batch_size=batch_size)

        next_boards = np.array([self.chess_brain.get(idx).next_board for idx in mini_batch])
        next_legal_moves = np.array([self.chess_brain.get(idx).next_legal_moves for idx in mini_batch])
        future_qs_list = target_model.predict([next_boards, next_legal_moves], batch_size=batch_size)
        
        X1 = []
        X2 = []
        Y = []
        # Optimize this loop
        for idx, replay_index in enumerate(mini_batch):
            memory =  self.chess_brain.get(replay_index)
            if not done:
                max_future_q = memory.reward + discount_factor * np.max(future_qs_list[idx])
            else:
                max_future_q = memory.reward

            current_qs = current_qs_list[idx]
            current_qs[memory.action] = (1 - learning_rate) * current_qs[memory.action] + learning_rate * max_future_q

            X1.append(memory.current_board)
            X2.append(memory.current_legal_moves)
            Y.append(current_qs)
        fit_obj = model.fit({'input_board': np.array(X1), 'input_legal_moves_mask':  np.array(X2)}, np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
        return fit_obj