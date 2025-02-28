import chess
import numpy as np

class ChessGame:
    """
    A Chess implementation for AlphaZero-style training.
    Uses python-chess for move validation and game logic.
    """

    def __init__(self):
        self.board = chess.Board()

    def getInitBoard(self):
        """
        Returns:
            board: Initial board state as a python-chess object.
        """
        return chess.Board()

    def getBoardSize(self):
        """
        Returns:
            (x, y): The dimensions of the board (8x8 for chess).
        """
        return (8, 8)

    def getActionSize(self):
        """
        Returns:
            actionSize: Total number of possible moves (moves encoded as an index).
        """
        return 4672  # Standard action space for chess in AlphaZero

    def getNextState(self, board, player, action):
        """
        Applies the given action to the board and returns the new board state.

        Returns:
            nextBoard (chess.Board() object)
            nextPlayer (-player)
        """
        next_board = board.copy()
        legal_moves = list(next_board.legal_moves)

        if action not in range(len(legal_moves)):  # Ensure action is within valid range
            raise ValueError(f"Invalid action {action}. Expected 0-{len(legal_moves)-1}, but got {action}")

        move = legal_moves[action]  # Direct lookup, no need to decode
        next_board.push(move)
        return next_board, -player

    def getValidMoves(self, board, player):
        """
        Returns a binary array indicating valid moves.

        Input:
            board: Current board state.
            player: Current player (1 or -1).

        Returns:
            validMoves: A binary vector of length self.getActionSize(),
                        with 1s for valid moves and 0s for invalid moves.
        """
        valid_moves = np.zeros(self.getActionSize(), dtype=np.uint8)
        legal_moves = list(board.legal_moves)  # Get legal moves list

        for idx, move in enumerate(legal_moves):  # Ensure valid index mapping
            valid_moves[idx] = 1  # Only mark existing legal moves as valid

        if np.sum(valid_moves) == 0:
            print(f"⚠️ WARNING: No valid moves for board state:\n{board}")

        return valid_moves

    def getGameEnded(self, board, player):
        """
        Determines the game outcome.

        Input:
            board: The current board state.
            player: The current player.

        Returns:
            0 if the game is ongoing.
            1 if the current player has won.
           -1 if the current player has lost.
            0.5 for a draw.
        """
        if board.is_checkmate():
            return 1 if board.turn != (player == 1) else -1
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0.5  # Draw
        return 0  # Game is still ongoing

    def getCanonicalForm(self, board, player):
        """
        Returns the board from the perspective of the given player.
        
        Input:
            board: The current chess.Board() object.
            player: 1 for White, -1 for Black.

        Returns:
            The board object without modifying it (since Chess is symmetrical in representation).
        """
        return board

    def getSymmetries(self, board, pi):
        """
        Generates symmetrical forms of the board and policy vector.

        Input:
            board: Current board state.
            pi: Policy vector.

        Returns:
            symmForms: List of (board, pi) tuples for training.
        """
        return [(board, pi)]  # No meaningful symmetrical transformations in chess.

    def stringRepresentation(self, board):
        """
        Provides a string representation of the board state.

        Input:
            board: Current board state.

        Returns:
            boardString: A string representation of the board.
        """
        return board.fen()  # Uses FEN (Forsyth-Edwards Notation) for easy board storage.

    def boardToMatrix(self, board):
        """
        Converts a chess.Board() object into a 6-channel 8x8 numerical representation.

        Args:
            board (chess.Board): The chess board to convert.

        Returns:
            np.ndarray: A 6-channel (6x8x8) matrix representation of the board.
        """
        piece_map = board.piece_map()
        matrix = np.zeros((6, 8, 8), dtype=np.float32)

        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type  # Pawn=1, Knight=2, etc.

            if piece.color == chess.WHITE:
                matrix[0][row][col] = piece_type  # White pieces in channel 0
            else:
                matrix[1][row][col] = piece_type  # Black pieces in channel 1

        # Additional feature planes
        matrix[2][:, :] = int(board.is_check() and board.turn == chess.WHITE)  # White in check
        matrix[3][:, :] = int(board.is_check() and board.turn == chess.BLACK)  # Black in check
        matrix[4][:, :] = board.fullmove_number / 100  # Normalize move count
        matrix[5][:, :] = 1 if board.turn == chess.WHITE else -1  # Player turn

        return matrix

    def encodeMove(self, move):
        """
        Encodes a chess move into an integer index.

        Input:
            move: A chess.Move object.

        Returns:
            move_index: Integer representation of the move (0 to 4671).
        """
        from_square = move.from_square
        to_square = move.to_square

        # Handle promotions (Queen=0, Rook=1, Bishop=2, Knight=3)
        promotion_offset = 0
        if move.promotion:
            promotion_offset = (move.promotion - 1) * 64

        move_index = from_square * 64 + to_square + promotion_offset

        return move_index if move_index < 4672 else None  # Ensure valid indexing

    def decodeMove(self, index, board):
        """
        Decodes an integer move index into a chess move.

        Input:
            index: Encoded move index.
            board: The current board state.

        Returns:
            move: A chess.Move object.
        """
        from_square = index // 64
        to_square = index % 64
        promotion = None

        # Handle promotions
        if index >= 4096:  # Promotion moves start at index 4096
            promotion_type = ((index - 4096) // 64) + 1
            promotion = chess.PieceType(promotion_type)

        move = chess.Move(from_square, to_square, promotion=promotion)
        if move in board.legal_moves:
            return move

        raise ValueError(f"Decoded move {move} is not legal on this board.")

    @staticmethod
    def display(board):
        """Prints a human-readable chess board."""
        print(board)
