import math
import numpy as np

EPS = 1e-8  # Small constant to prevent division by zero

class MCTS:
    """
    Monte Carlo Tree Search for AlphaZero Chess.
    """

    def __init__(self, game, nnet, args):
        """
        Initializes MCTS with game logic, neural network, and search parameters.
        """
        self.game = game  # ChessGame instance
        self.nnet = nnet  # ChessNeuralNet instance
        self.args = args  # Configuration parameters

        # MCTS statistics
        self.Qsa = {}  # Q-values for state-action pairs (expected reward)
        self.Nsa = {}  # Number of times an action was taken from a state
        self.Ns = {}   # Number of times a state was visited
        self.Ps = {}   # Policy predictions from the neural network

        # Game state caches
        self.Es = {}  # Stores game results (-1, 0, 1)
        self.Vs = {}  # Stores valid moves for each state

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Runs multiple MCTS simulations and returns move probabilities.

        Input:
            canonicalBoard: The board from the current player's perspective.
            temp: Temperature parameter for exploration-exploitation tradeoff.

        Returns:
            probs: A probability distribution over possible actions.
        """
        for _ in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())])

        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros(len(counts))
            probs[best_action] = 1
            return probs

        # Softmax over visit counts (with temperature scaling)
        counts = counts ** (1. / temp)
        counts_sum = np.sum(counts)
        probs = counts / counts_sum if counts_sum > 0 else np.zeros_like(counts)
        return probs

    def search(self, canonicalBoard):
        """
        Recursively performs MCTS search.

        Input:
            canonicalBoard: The board from the current player's perspective.

        Returns:
            v: The negative of the estimated value of the state.
        """
        s = self.game.stringRepresentation(canonicalBoard)

        # If the game has ended, return the outcome
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            print(f"Terminal state reached for board {s} with outcome {self.Es[s]}")
            return -self.Es[s]

        # If the state is not in the tree, expand it
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(self.game.getCanonicalForm(canonicalBoard, 1))
            valids = self.game.getValidMoves(canonicalBoard, 1)

            if np.sum(valids) > 0:
                self.Ps[s] = self.Ps[s] * valids  # Keep only valid moves
                self.Ps[s] /= np.sum(self.Ps[s])  # Normalize so probabilities sum to 1
            else:
                print(f"WARNING: No valid moves available for board state {s}. Using uniform distribution.")
                self.Ps[s] = valids / np.sum(valids)  # Assign equal probability if all moves are masked

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v  # Return negative value for the opponent

        # Select action using Upper Confidence Bound (UCT)
        valid_moves = np.where(self.Vs[s] == 1)[0]  # Get indices of valid moves
        if len(valid_moves) == 0:
            raise ValueError(f"No valid moves available at board state: {s}")

        #print(f"Valid moves indices: {valid_moves}")  # Debugging line

        best_action, best_uct = None, -float("inf")

        for a in valid_moves:  # Iterate only over legal move indices
            if (s, a) in self.Qsa:
                uct = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                uct = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Assume Q(s,a) = 0 initially

            #print(f"Evaluating move {a}: UCT = {uct}")  # Debugging line

            if uct > best_uct:
                best_uct = uct
                best_action = a

        if best_action is None:
            raise ValueError(f"MCTS failed to find a valid action. Board state: {s}")

        #print(f"Selected action: {best_action}")  # Debugging line

        # Get the next state
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, best_action)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        # Recursively search the new state
        v = self.search(next_s)

        # Update Q-values and visit counts
        if (s, best_action) in self.Qsa:
            self.Qsa[(s, best_action)] = (self.Nsa[(s, best_action)] * self.Qsa[(s, best_action)] + v) / (self.Nsa[(s, best_action)] + 1)
            self.Nsa[(s, best_action)] += 1
        else:
            self.Qsa[(s, best_action)] = v
            self.Nsa[(s, best_action)] = 1

        self.Ns[s] += 1
        return -v  # Return negative value for the opponent
