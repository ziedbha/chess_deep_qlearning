import numpy as np
import chess
import chess.engine
import tensorflow as tf
import chess.svg

rewards = { 'loss': -1000, 'win_10':1000,  'win_20':500, 'win_50':100, 'win_100':50, 'win_other':25,
            'pawn':1, 'knight':3,'bishop':3.5,'rook':5,'queen':9,'en_passant':1}

# def get_max_reward():
#     return rewards['win_10'] + 8 * rewards['pawn'] + 2 * (rewards['rook'] + rewards['bishop'] + rewards['knight']) + rewards['queen']

# def get_min_reward():
#     return rewards['loss'] - 8 * rewards['pawn'] - 2 * (rewards['rook'] + rewards['bishop'] + rewards['knight']) - rewards['queen']

# MAX_REWARD = get_max_reward()
# MIN_REWARD = get_min_reward() - 1

# REWARD_RANGE = MAX_REWARD - MIN_REWARD

# def normalize_rewards_in_sigmoid_range():
#     for reward in rewards:
#         rewards[reward] = (rewards[reward] + (-MIN_REWARD)) / REWARD_RANGE

# normalize_rewards_in_sigmoid_range()

ILLEGAL_ACTION_LOGITS_PENALTY = -1e6

# Theoretically: 64x64 from->to moves, ((8 non-capture promotions, 6*2 capture promotions, 2 edge capture promotions) to queen or knight, for white and black)
num_actions = 0

num2move = {}
move2num = {}
counter = 0

# Possible moves
for from_sq in range(64):
    for to_sq in range(64):
        num2move[counter] = chess.Move(from_sq,to_sq)
        move2num[chess.Move(from_sq,to_sq)] = counter
        counter += 1

# Possible promotions (non-capture)
for i in range(8):
    black_from_sq = i + 1 * 8
    black_to_sq = i

    white_from_sq = i + 8 * 6
    white_to_sq = i + 8 * 7

    q_black = chess.Move(black_from_sq, black_to_sq, chess.QUEEN)
    q_white = chess.Move(white_from_sq, white_to_sq, chess.QUEEN)
    k_black = chess.Move(black_from_sq, black_to_sq, chess.KNIGHT)
    k_white = chess.Move(white_from_sq, white_to_sq, chess.KNIGHT)

    num2move[counter] = q_black
    move2num[q_black] = counter
    counter += 1

    num2move[counter] = q_white
    move2num[q_white] = counter
    counter += 1

    num2move[counter] = k_black
    move2num[k_black] = counter
    counter += 1

    num2move[counter] = k_white
    move2num[k_white] = counter
    counter += 1

# Black capture promotions
bound_low = 8 * 0 # back rank start
bound_high = bound_low + 8 - 1 # back rank end
for i in range(8):
    black_from_sq = i + 1 * 8
    for offset in [-1, 1]:
        black_to_sq = i + offset
        if (black_to_sq >= bound_low and black_to_sq <= bound_high):
            k_black = chess.Move(black_from_sq, black_to_sq, chess.KNIGHT)
            q_black = chess.Move(black_from_sq, black_to_sq, chess.QUEEN)

            num2move[counter] = k_black
            move2num[k_black] = counter
            counter += 1

            num2move[counter] = q_black
            move2num[q_black] = counter
            counter += 1

# White capture promotions
bound_low = 8 * 7 # back rank start
bound_high = bound_low + 8 - 1 # back rank end
for i in range(8):
    white_from_sq = i + 8 * 6
    for offset in [-1, 1]:
        white_to_sq = i + 8 * 7 + offset
        if (white_to_sq >= bound_low and white_to_sq <= bound_high):
            k_white = chess.Move(white_from_sq, white_to_sq, chess.KNIGHT)
            q_white = chess.Move(white_from_sq, white_to_sq, chess.QUEEN)

            num2move[counter] = k_white
            move2num[k_white] = counter
            counter += 1

            num2move[counter] = q_white
            move2num[q_white] = counter
            counter += 1

num_actions = len(move2num)

# Use this to debug a board if needed
example_board = chess.Board('5bnr/4k2P/6KP/p1r5/P7/8/8/8 w - - 1 145')

# Filter legal moves
def filter_legal_moves(board):
    filter_mask =  np.ones(shape=(num_actions)) * ILLEGAL_ACTION_LOGITS_PENALTY
    # Fast but inaccurate
    # legal_moves = list(board.legal_moves)
    # for legal_move in legal_moves:
    #     from_square = legal_move.from_square
    #     to_square = legal_move.to_square
    #     idx = move2num[chess.Move(from_square,to_square)]
    #     filter_mask[idx] = 0

    # Super slow brute force
    # for i in range(num_actions):
    #     if board.is_legal(num2move[i]):
    #         filter_mask[i] = 0

    #Middle ground
    pieces = [chess.PAWN, chess.KNIGHT, chess.KING, chess.QUEEN, chess.BISHOP, chess.ROOK]
    colors = [chess.WHITE, chess.BLACK]
    for piece in pieces:
        for color in colors:
            from_squares = board.pieces(piece, color)
            for from_square in from_squares:
                for to_square in range(64):
                    move = chess.Move(from_square, to_square)
                    if board.is_legal(move):
                        filter_mask[move2num[move]] = 0

    # Promotion moves
    for move_idx in range(64 * 64, num_actions):
        if board.is_legal(num2move[move_idx]):
            filter_mask[move_idx] = 0
    return filter_mask

def translate_board(board): 
    pgn = board.epd()
    foo = []  
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(chess_dict['.'])
            else:
                foo2.append(chess_dict[thing])
        foo.append(foo2)
    return np.array(foo)

#Uppercase => White
#Lowercase => Black
chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

#Environement
class ChessEnv():
    def __init__(self):
        self.board = chess.Board()
        pass

    def get_legal_moves_mask(self):
        return filter_legal_moves(self.board)

    # Translate board state into 1 hot encodings
    def translate_board(self):
        return translate_board(self.board)

    # Reset board state
    def reset(self):
        self.board = chess.Board()
        legal_moves = self.get_legal_moves_mask()
        return [np.array(translate_board(self.board)), legal_moves]

    def get_game_moves(self):
        return self.board.move_stack

    def checkmate_reward(self):
        num_turns = self.board.fullmove_number * 2
        if num_turns < 10:
            return rewards['win_10']
        if num_turns < 20:
            return rewards['win_20']
        if num_turns < 50:
            return rewards['win_50']
        if num_turns < 100:
            return rewards['win_100']
        return rewards['win_other']

    def get_board(self):
        return self.board
    
    # Advance board state by taking an action
    def step(self, action,  target_model, model, is_source_model = True):
        reward = 0
        done = False
        is_checkmate = False
        is_capture = False

        # Debug illegal moves
        if not self.board.is_legal(action):
            if not is_source_model:
                target_model.predict_and_pick_best(self)
            else:
                model.predict_and_pick_best(self)
            legal_moves = filter_legal_moves(self.board)
            print("Illegal move!")

        # If move is a promotion move
        if action.promotion:
            reward += 5

        # Rewards for taking pieces
        if self.board.is_capture(action):
            is_capture = True
            if self.board.is_en_passant(action):
                reward += rewards['en_passant']
            else:
                piece_type = self.board.piece_at(action.to_square).piece_type
                if piece_type == chess.PAWN:
                    reward += rewards['pawn']
                elif piece_type == chess.ROOK:
                    reward += rewards['rook']
                elif piece_type == chess.BISHOP:
                    reward += rewards['bishop']
                elif piece_type == chess.KNIGHT:
                    reward += rewards['knight']
                elif piece_type == chess.QUEEN:
                    reward += rewards['queen']
                else:
                    raise ValueError('Illegal capture!')

        # Advance board state
        self.board.push(action)
        next_board = translate_board(self.board)
        next_legal_moves = self.get_legal_moves_mask()
        
        if self.board.is_checkmate():
            is_checkmate = True
            reward = self.checkmate_reward()

        if self.board.is_game_over():
            done = True

        # Negative reward if not source_model
        if not is_source_model:
            if is_checkmate:
                reward = rewards['loss']
            else:
                reward = reward * -1
        
        return [next_board, next_legal_moves], reward, done, is_checkmate, is_capture
