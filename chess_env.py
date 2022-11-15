import numpy as np
import chess
import chess.engine
import tensorflow as tf

REWARD_OFFSET = 140

num_actions = 4096

# Filter legal moves
def filter_legal_moves(board):
    filter_mask =  np.zeros(shape=(num_actions))
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        idx = move2num[chess.Move(from_square,to_square)]
        filter_mask[idx] = 1
    return filter_mask

num2move = {}
move2num = {}
counter = 0

# Possible modes
for from_sq in range(64):
    for to_sq in range(64):
        num2move[counter] = chess.Move(from_sq,to_sq)
        move2num[chess.Move(from_sq,to_sq)] = counter
        counter += 1

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
    
    # Advance board state by taking an action
    def step(self, action, is_source_model = True):
        reward = 0
        done = False
        is_checkmate = False

        # Rewards for taking pieces
        if self.board.is_capture(action):
            piece_type = self.board.piece_at(action.to_square).piece_type
            if piece_type == chess.PAWN:
                reward = 1
            if piece_type == chess.ROOK:
                reward = 5
            if piece_type == chess.BISHOP:
                reward = 3.5
            if piece_type == chess.KNIGHT:
                reward = 3
            if piece_type == chess.QUEEN:
                reward = 9

        # Advance board state
        self.board.push(action)
        next_board = translate_board(self.board)
        next_legal_moves = self.get_legal_moves_mask()
        
        if self.board.is_checkmate():
            is_checkmate = True
            reward = 100

        if self.board.is_game_over():
            done = True

        # Negative reward if not source_model
        if not is_source_model:
            reward = reward * -1

        reward += REWARD_OFFSET
        
        return [next_board, next_legal_moves], reward, done, is_checkmate
