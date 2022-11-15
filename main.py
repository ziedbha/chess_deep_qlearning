from chess_env import *
from q_model import *
import datetime
import pickle
import os

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

env = ChessEnv()
model = Q_model()
target_model = Q_model()
model.model.summary()
train = True

def dump_game(moves, episode, iter, result):
    filename = './games/episode_' + str(episode) + '_iter_' + str(iter) + '_' + result + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(moves, f)

def replay_game(filename):
    with open(filename, 'rb') as f:
        move_stack = pickle.load(f)

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
    decay = 0.01

    # 1. Initialize the Target and Main models
    target_model.model.set_weights(model.model.get_weights())

    # X = states, y = actions
    X = []
    y = []
    steps_to_update_target_model = 0

    # An episode is a full game
    train_episodes = 1000

    num_trained_frames = 0

    wins = 0
    losses = 0
    draws = 0

    # Have we updated the network weights since the last time we dumped to disk?
    updates_since_dump = 1

    for episode in range(train_episodes):
        total_training_rewards = 0
        state = env.reset()
        done = False
        iterations_in_episode = 0
        while not done:
            iterations_in_episode += 1
            steps_to_update_target_model += 1
            random_number = np.random.rand()

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                chess_move, action = model.explore(env, False)
            else:
                # Exploit best known action
                chess_move, action = model.predict_and_pick_best(env)

            # Step the environment using chosen action
            is_win = False
            is_loss = False
            is_draw = False
            new_state, reward, done, is_win = env.step(chess_move, True)
            total_training_rewards += reward - REWARD_OFFSET
            if (is_win):
                dump_game(env.get_game_moves(), episode, steps_to_update_target_model, 'win')
                wins += 1

            # Jump to adversary's turn if possible
            if not done:
                chess_move_other, action_other = target_model.predict_and_pick_best(env)
                new_state, reward_other, done, is_loss = env.step(chess_move_other, False)
                reward += reward_other
                total_training_rewards += reward_other - REWARD_OFFSET
                if (is_loss):
                    dump_game(env.get_game_moves(), episode, steps_to_update_target_model, 'loss')
                    losses += 1

            # Record if draw
            if done and (not is_loss) and (not is_win):
                is_draw = True
                draws += 1
            
            # Record the new memory
            model.chess_brain.add(ChessBrain.ChessMemory(state[0], state[1], new_state[0], new_state[1], action, reward, done))
            
            # Board state moves to new state
            state = new_state

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                training_history = model.train(model.model, target_model.model, done)
                if training_history is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', training_history.history['loss'][0], step=num_trained_frames)
                    num_trained_frames += 1

            # 4. Game is done, possibly record weights
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
    source_episode = './checkpoints/chess_model_weights_episode_20.h5'
    target_episode = './checkpoints/chess_model_weights_episode_0.h5'
    model.model.load_weights(source_episode)
    target_model.model.load_weights(target_episode)

    test_episodes = 4000
    wins = 0
    draws = 0
    losses = 0
    games_played = 0

    for episode in range(test_episodes):
        done = False
        source_turn = True
        games_played += 1
        while not done:
            chess_move = None
            action = None
            if source_turn:
                chess_move, action = model.predict_and_pick_best(env)
            else:
                chess_move, action = target_model.predict_and_pick_best(env)

            source_turn = not source_turn
 
            _, reward, done, _ = env.step(chess_move, source_turn)
            if done:
                if reward == 100:
                    wins += 1
                    print("Checkmate! win.")
                elif reward == -100:
                    losses += 1
                    print("Checkmate.. loss.")
                else:
                    draws += 1
                    print("Draw.")            
    print('Test results: {} wins, {} losses, {} draws'.format(wins, losses, draws))