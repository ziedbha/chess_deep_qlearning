from chess_env import *
from q_model import *
import datetime
from collections import deque


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

env = ChessEnv()

model = Q_model()
target_model = Q_model()

model.model.summary()

train = True

if train:
    np.random.seed(42)
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    # 1. Initialize the Target and Main models
    target_model.model.set_weights(model.model.get_weights())
    replay_memory = deque(maxlen=100_000)

    # X = states, y = actions
    X = []
    y = []
    steps_to_update_target_model = 0

    # An episode a full game
    train_episodes = 300

    num_trained_frames = 0

    env.step(chess.Move.from_uci('e1e2'))

    for episode in range(train_episodes):
        total_training_rewards = 0
        state = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()

            # 2. Explore using the Epsilon Greedy Exploration Strategy
            if random_number <= epsilon:
                # Explore
                chess_move, action = model.explore(env, False)
            else:
                # Exploit best known action
                chess_move, action = model.predict_and_pick_best(env)

            new_state, reward, done = env.step(chess_move, True)
            total_training_rewards += reward - REWARD_OFFSET

            # Jump to adversary's turn if possible
            if not done:
                chess_move_other, action_other = target_model.predict_and_pick_best(env)
                new_state, reward_other, done = env.step(chess_move_other, False)
                reward += reward_other
                total_training_rewards += reward_other - REWARD_OFFSET
            
            # Record the new memory
            replay_memory.append([state, action, reward, new_state, done])
            
            # Board state moves to new state
            state = new_state

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                training_history = model.train(replay_memory, model.model, target_model.model, done)
                if training_history is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', training_history.history['loss'][0], step=num_trained_frames)
                    num_trained_frames += 1

            if done:
                print('Total training rewards this episode: {} after n steps = {}'.format(total_training_rewards, episode))

                if steps_to_update_target_model > 500:
                    print('Copying main network weights to the target network weights')
                    target_model.model.set_weights(model.model.get_weights())
                    steps_to_update_target_model = 0

                    if episode % 10 == 0:
                        filename = 'chess_model_weights_episode_' + str(episode) + '.h5'
                        target_model.model.save_weights(filename)
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)
else:
    source_episode = 'chess_model_weights_episode_20.h5'
    target_episode = 'chess_model_weights_episode_0.h5'
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

            _, reward, done = env.step(chess_move, source_turn)
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