import numpy as np
import chess
import chess.engine
import tensorflow as tf
import chess.svg

MAX_REWARD = 1000 + 9 + 1 * 12 + 3.5 * 2 + 5 * 2 + 3 * 2

num_actions = 64 * 64 + 16 + 16

# Filter legal moves
def filter_legal_moves(board):
    filter_mask =  np.ones(shape=(num_actions)) * (-MAX_REWARD)
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

num2move = {}
move2num = {}
counter = 0

# Possible moves
for from_sq in range(64):
    for to_sq in range(64):
        num2move[counter] = chess.Move(from_sq,to_sq)
        move2num[chess.Move(from_sq,to_sq)] = counter
        counter += 1

# Possible promotions
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
            return 1000
        if num_turns < 50:
            return 500
        if num_turns < 100:
            return 100
        if num_turns < 150:
            return 75
        if num_turns < 200:
            return 50
        return 20

    def get_board(self):
        return self.board
    
    # Advance board state by taking an action
    def step(self, action, is_source_model = True):
        reward = 0
        done = False
        is_checkmate = False

        if not self.board.is_legal(action):
            legal_moves = filter_legal_moves(self.board)
            print("Illegal move!")

        # If move is a promotion move
        if action.promotion:
            reward += 5

        # Rewards for taking pieces
        if self.board.is_capture(action):
            if self.board.is_en_passant(action):
                reward += 1
            else:
                piece_type = self.board.piece_at(action.to_square).piece_type
                if piece_type == chess.PAWN:
                    reward += 1
                if piece_type == chess.ROOK:
                    reward += 5
                if piece_type == chess.BISHOP:
                    reward += 3.5
                if piece_type == chess.KNIGHT:
                    reward += 3
                if piece_type == chess.QUEEN:
                    reward += 9

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
                reward = -1000
            else:
                reward = reward * -1
        
        return [next_board, next_legal_moves], reward, done, is_checkmate
