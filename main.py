from chess_env import *
from q_model import *
import datetime
import pickle
import os

# TODOs:
# Hyper-pramater tuning:
#   Implement learning rate decay
#   Lower learning rate decay
#   Test on 500 games each time
# Do a lot more exploration
# Change to pytorch

# Make default dump directories
DEFAULT_WEIGHT_FILENAME = "chess_model_weights_episode_"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_GAME_DUMP_DIR = "./games"

def dump_game(moves, episode, iter, is_white, result):
    player_id = "white"
    if not is_white:
        player_id = "black"
    filename = DEFAULT_GAME_DUMP_DIR + '/episode_' + str(episode) + '_iter_' + str(iter) + '_' + player_id + '_' + result + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(moves, f)

def play_source_model_turn(epsilon, chess_env, model):
    # 1. Explore using the Epsilon Greedy Exploration Strategy
    random_number = np.random.rand()
    if random_number <= epsilon:
        # Explore
        chess_move, action = model.explore(chess_env, False)
    else:
        # Exploit best known action
        chess_move, action = model.predict_and_pick_best(chess_env)

    # 2.a. Step the environment using chosen action
    new_state, reward, done, is_win, is_capture = chess_env.step(chess_move, True)
    return new_state, action, reward, done, is_win, is_capture

def play_target_model_turn(chess_env, target_model):
    # Target model picks the best move it thinks about
    chess_move_other, _ = target_model.predict_and_pick_best(chess_env)
    new_state, reward_other, done, is_loss, is_capture = chess_env.step(chess_move_other, False)
    # is_loss is relative to the source model
    return new_state, reward_other, done, is_loss, is_capture

def train_model(train_params, tensorboard):
    np.random.seed(42)

    env = ChessEnv()
    model = Q_model()
    target_model = Q_model()
    model.model.summary()
    model.change_optimizer(train_params['learning_rate'])
    model.change_bellman_learning_rate(train_params['bellman_learning_rate'])

    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # 1. Initialize the Target and Main models
    target_model.model.set_weights(model.model.get_weights())

    # X = states, y = actions
    steps_to_update_target_model = 0

    # An episode is a full game
    num_trained_frames = 0
    wins = 0
    losses = 0
    draws = 0

    # Load old model
    if train_params['load_previous_checkpoint']:
        model.model.load_weights(train_params['old_model_weights'])
        target_model.model.load_weights(train_params['old_model_weights'])

    # Have we updated the network weights since the last time we dumped to disk?
    updates_since_dump = 1

    # Is source model playing white?
    source_goes_first = False

    total_training_rewards = 0
    for episode in range(train_params['start_episode'], train_params['num_episodes'] + 1):
        total_episode_rewards = 0
        state = env.reset()
        done = False
        iterations_in_episode = 0

        # Swap turns so that network learns both white and black sides
        source_goes_first = not source_goes_first
        if (not source_goes_first):
            # Game can never end on turn 1
            play_target_model_turn(env, target_model)
     
        while not done:
            iterations_in_episode += 1
            steps_to_update_target_model += 1

            is_win = False
            is_loss = False
            is_draw = False
            is_capture = False

            # 1.a. Play source turn
            new_state, action, reward, done, is_win, is_capture = play_source_model_turn(epsilon, env, model)
            total_episode_rewards += reward

            # 1.b. Play target turn if game is not over yet
            if not done:
                new_state, reward, done, is_loss, is_capture = play_target_model_turn(env, target_model)
                total_episode_rewards += reward
                    
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
                training_history = model.train(target_model.model, done)
                if training_history is not None:
                    with tensorboard.as_default():
                        tf.summary.scalar('loss', training_history.history['loss'][0], step=num_trained_frames)
                    num_trained_frames += 1

            # 5. Game is done, possibly record new weights
            if done:
                print('Total training rewards this episode = {} after n steps = {}.'.format(total_episode_rewards, episode))
                if (is_win):
                    print('Won game!')

                if (is_loss):
                    print('Lost game....')

                if (is_draw):
                    print('Draw.')

                with tensorboard.as_default():
                    total_training_rewards += total_episode_rewards
                    tf.summary.scalar('reward', total_training_rewards, step=episode)

                is_last_episode = episode == train_params['num_episodes']

                if steps_to_update_target_model > 500 or is_last_episode:
                    print('Copying main network weights to the target network weights')
                    target_model.model.set_weights(model.model.get_weights())
                    steps_to_update_target_model = 0
                    updates_since_dump += 1

                if (episode % 10 == 0 and updates_since_dump != 0) or is_last_episode:
                    updates_since_dump = 0
                    filename = DEFAULT_CHECKPOINT_DIR + '/' + DEFAULT_WEIGHT_FILENAME + str(episode) + '.h5'
                    target_model.model.save_weights(filename)
                break
        epsilon = train_params['min_epsilon'] + (train_params['max_epsilon'] - train_params['min_epsilon']) * np.exp(-train_params['epsilon_decay'] * episode)
    model.unload()
    target_model.unload()
    
def test_model(test_params, tensorboard):
    np.random.seed(42)

    env = ChessEnv()
    model = Q_model()
    target_model = Q_model()
    model.model.summary()

    model.model.load_weights(DEFAULT_CHECKPOINT_DIR + '/' + DEFAULT_WEIGHT_FILENAME + str(test_params['source_episode']) + '.h5')
    if not test_params['random_opponent']:
        target_model.model.load_weights(DEFAULT_CHECKPOINT_DIR + DEFAULT_WEIGHT_FILENAME + str(test_params['target_episode']) + '.h5')

    wins = 0
    draws = 0
    losses = 0
    games_played = 0
    was_white = True

    for episode in range(test_params['num_episodes']):
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
            
            _, _, done, is_checkmate, _ = env.step(chess_move, source_turn)
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
    model.unload()
    target_model.unload()
    return wins, losses, draws

def tune_learning_rates(train_params, test_params):
    bellman_alphas = [0.9, 0.5, 0.1, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    weight_alphas = bellman_alphas

    fixed_weight_alpha = 0.1
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    validation_scores = []
    for bellman_alpha in bellman_alphas:
        # Create tensorboard logs
        train_log_dir = 'logs/' + current_time + '/train' + '_alpha_' + str(fixed_weight_alpha) + "_bellman_alpha_" + str(bellman_alpha)
        test_log_dir = 'logs/' + current_time + '/test' + '_alpha_' + str(fixed_weight_alpha) + "_bellman_alpha_" + str(bellman_alpha)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        train_params['bellman_learning_rate'] = bellman_alpha
        train_params['learning_rate'] = fixed_weight_alpha  
        train_model(train_params, train_summary_writer)

        test_params['source_episode'] = train_params['num_episodes']
        wins, losses, draws = test_model(test_params, test_summary_writer)
        validation_scores.append({'wins': wins, 'losses': losses, 'draws': draws})

    max_wins = 0
    max_losses = 0
    max_wins_idx = 0
    max_losses_idx = 0
    print("-----------------------------")
    for idx, bellman_alpha in enumerate(bellman_alphas):
        if validation_scores[idx]['wins'] > max_wins:
            max_wins = validation_scores[idx]['wins']
            max_wins_idx = idx

        if validation_scores[idx]['losses'] > max_losses:
            max_losses = validation_scores[idx]['losses']
            max_losses_idx = idx

        print('Bellman alpha {} eval scores: wins {}, losses {}, draws {}.'.format(bellman_alpha,
                                                                                    validation_scores[idx]['wins'],
                                                                                    validation_scores[idx]['losses'],
                                                                                    validation_scores[idx]['draws']))
    print("-----------------------------")
    print('Best bellman alpha is {}'.format(bellman_alphas[max_wins_idx]))
    print('Worst bellman alpha is {}'.format(bellman_alphas[max_losses_idx]))

if __name__=="__main__":
    if not os.path.exists(DEFAULT_CHECKPOINT_DIR):
        os.makedirs(DEFAULT_CHECKPOINT_DIR)

    if not os.path.exists(DEFAULT_GAME_DUMP_DIR):
        os.makedirs(DEFAULT_GAME_DUMP_DIR)

    train_params = {'start_episode':0, 'load_previous_checkpoint':False, 'old_model_weights': "", 
                'learning_rate':1e-4, 'epsilon_decay':0.005, 'num_episodes':100, 'bellman_learning_rate': 0.5,
                'min_epsilon': 0.01, 'max_epsilon': 0.1}

    test_params = {'source_episode': "", 'target_episode': "", 'random_opponent': True, 'num_episodes': 150}

    tune_learning_rates(train_params, test_params)