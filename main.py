from chess_env import *
from q_model import *
import datetime
import pickle
import os

# TODOs:
# Hyper-pramater tuning:
#   Implement learning rate decay
#   Lower learning rate decay
#   Test on 500 games each time, with random agent
# Do a lot more exploration
# Change to pytorch

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

env = ChessEnv()
model = Q_model()
target_model = Q_model()
model.model.summary()

# Training parameters
train = False
load_previous_checkpoint = False
start_episode = 90
old_model_weights = './checkpoints/chess_model_weights_episode_90.h5'

# Testing parameters
source_episode = './checkpoints/chess_model_weights_episode_90.h5'
target_episode = './checkpoints/chess_model_weights_episode_0.h5'
random_opponent = True

def dump_game(moves, episode, iter, is_white, result):
    player_id = "white"
    if not is_white:
        player_id = "black"
    filename = './games/episode_' + str(episode) + '_iter_' + str(iter) + '_' + player_id + '_' + result + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(moves, f)

def play_source_model_turn(epsilon):
    # 1. Explore using the Epsilon Greedy Exploration Strategy
    random_number = np.random.rand()
    if random_number <= epsilon:
        # Explore
        chess_move, action = model.explore(env, False)
    else:
        # Exploit best known action
        chess_move, action = model.predict_and_pick_best(env)

    # 2.a. Step the environment using chosen action
    new_state, reward, done, is_win, is_capture = env.step(chess_move,target_model, model, True)
    return new_state, action, reward, done, is_win, is_capture

def play_target_model_turn():
    # Target model picks the best move it thinks about
    chess_move_other, _ = target_model.predict_and_pick_best(env)
    new_state, reward_other, done, is_loss, is_capture = env.step(chess_move_other, target_model, model, False)
    # is_loss is relative to the source model
    return new_state, reward_other, done, is_loss, is_capture
    
if train:
    np.random.seed(42)

    # Make dump directories
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    if not os.path.exists("./games"):
        os.makedirs("./games")

    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.0005

    # 1. Initialize the Target and Main models
    target_model.model.set_weights(model.model.get_weights())

    # X = states, y = actions
    steps_to_update_target_model = 0

    # An episode is a full game
    train_episodes = 5000
    num_trained_frames = 0
    wins = 0
    losses = 0
    draws = 0

    # Load old model
    if load_previous_checkpoint:
        model.model.load_weights(old_model_weights)
        target_model.model.load_weights(old_model_weights)
    else:
        start_episode = 0

    # Have we updated the network weights since the last time we dumped to disk?
    updates_since_dump = 1

    # Is source model playing white?
    source_goes_first = False

    for episode in range(start_episode, train_episodes):
        total_training_rewards = 0
        state = env.reset()
        done = False
        iterations_in_episode = 0

        # Swap turns so that network learns both white and black sides
        source_goes_first = not source_goes_first
        if (not source_goes_first):
            # Game can never end on turn 1
            play_target_model_turn()
     
        while not done:
            iterations_in_episode += 1
            steps_to_update_target_model += 1

            is_win = False
            is_loss = False
            is_draw = False
            is_capture = False

            # 1.a. Play source turn
            new_state, action, reward, done, is_win, is_capture = play_source_model_turn(epsilon)
            total_training_rewards += reward

            # 1.b. Play target turn if game is not over yet
            if not done:
                new_state, reward, done, is_loss, is_capture = play_target_model_turn()
                total_training_rewards += reward
                    
            # 2. Record end game state 
            if done:
                if (is_win):
                    dump_game(env.get_game_moves(), episode, iterations_in_episode, source_goes_first, 'win')
                    wins += 1
                elif is_loss:
                    dump_game(env.get_game_moves(), episode, iterations_in_episode, source_goes_first, 'loss')
                    losses += 1
                else:
                    is_draw = True
                    draws += 1
            
            # 3. Record the new memory
            model.chess_brain.add(ChessBrain.ChessMemory(state[0], state[1], new_state[0], new_state[1], action, reward, done, (is_win or is_loss), is_capture))
            
            # Board state moves to new state
            state = new_state

            # 4. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                training_history = model.train(model.model, target_model.model, done)
                if training_history is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', training_history.history['loss'][0], step=num_trained_frames)
                    num_trained_frames += 1

            # 5. Game is done, possibly record new weights
            if done:
                print('Total training rewards this episode = {} after n steps = {}.'.format(total_training_rewards, episode))
                if (is_win):
                    print('Won game!')

                if (is_loss):
                    print('Lost game....')

                if (is_draw):
                    print('Draw.')

                if steps_to_update_target_model > 500:
                    print('Copying main network weights to the target network weights')
                    target_model.model.set_weights(model.model.get_weights())
                    steps_to_update_target_model = 0
                    updates_since_dump += 1

                if episode % 10 == 0 and updates_since_dump != 0:
                    updates_since_dump = 0
                    filename = './checkpoints/chess_model_weights_episode_' + str(episode) + '.h5'
                    target_model.model.save_weights(filename)
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
else:
    model.model.load_weights(source_episode)
    if not random_opponent:
        target_model.model.load_weights(target_episode)

    test_episodes = 50
    wins = 0
    draws = 0
    losses = 0
    games_played = 0
    was_white = True

    for episode in range(test_episodes):
        done = False
        source_turn = was_white
        turns = 0
        env.reset()
        while not done:
            chess_move = None
            if source_turn:
                chess_move, _ = model.predict_and_pick_best(env)
            else:
                chess_move, _ = target_model.explore(env, False)
            
            _, _, done, is_checkmate, _ = env.step(chess_move, target_model, model, source_turn)
            if done:
                if is_checkmate and source_turn:
                    wins += 1
                    print("Checkmate! win.")
                    dump_game(env.get_game_moves(), episode, turns, was_white, "win_test")
                elif is_checkmate and not source_turn:
                    losses += 1
                    print("Checkmate.. loss.")
                    dump_game(env.get_game_moves(), episode, turns, was_white, "loss_test")
                else:
                    draws += 1
                games_played += 1
                was_white = not was_white
                print("Game {} ended after {} turns.".format(games_played, turns))
            
            turns += 1
            source_turn = not source_turn
    print('Test results: {} wins, {} losses, {} draws'.format(wins, losses, draws))