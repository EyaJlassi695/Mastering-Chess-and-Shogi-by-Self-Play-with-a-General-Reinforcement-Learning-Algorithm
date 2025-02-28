import logging
from tqdm import tqdm
import chess

log = logging.getLogger(__name__)

class Arena:
    """
    An Arena class where two agents can be pitted against each other in Chess.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Args:
            player1, player2: Functions that take a board as input and return an action.
            game: ChessGame instance.
            display: Function to print the board (optional).
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display  # If provided, this function will print the board.

    def playGame(self, verbose=False):
        """
        Plays a single game between player1 and player2.

        Returns:
            1  → player1 wins
           -1  → player2 wins
            0  → draw
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0

        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose and self.display:
                print(f"Turn {it}, Player {curPlayer}")
                self.display(board)

            # Get the move from the current player
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            # Validate the move
            valid_moves = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valid_moves[action] == 0:
                log.error(f'Invalid move: {action}')
                assert valid_moves[action] > 0  # Ensure move is legal

            # Apply the move
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

        # Game ended, determine the result
        result = self.game.getGameEnded(board, 1)

        if verbose and self.display:
            print(f"Game Over: Turn {it}, Result: {result}")
            self.display(board)

        return result

    def playGames(self, numGames, verbose=False):
        """
        Plays multiple games and tracks wins/losses.

        Args:
            numGames: Total number of games to play.
            verbose: If True, prints game progress.

        Returns:
            (player1 wins, player2 wins, draws)
        """
        numGames = numGames // 2  # Half the games start with player1, half with player2
        oneWon, twoWon, draws = 0, 0, 0

        for _ in tqdm(range(numGames), desc="Arena: Player1 starts"):
            result = self.playGame(verbose)
            if result == 1:
                oneWon += 1
            elif result == -1:
                twoWon += 1
            else:
                draws += 1

        # Swap players and repeat
        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(numGames), desc="Arena: Player2 starts"):
            result = self.playGame(verbose)
            if result == -1:
                oneWon += 1
            elif result == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

