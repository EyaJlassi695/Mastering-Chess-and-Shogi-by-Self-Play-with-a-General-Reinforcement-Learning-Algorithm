import os
import sys
import logging
import numpy as np
from collections import deque
from random import shuffle
from pickle import Pickler, Unpickler
from tqdm import tqdm

from MCTS import MCTS

log = logging.getLogger(__name__)

class ChessCoach:
    """
    Manages self-play training for AlphaZero Chess.
    """

    def __init__(self, game, nnet, args):
        """
        Initializes the coach with the chess game, neural network, and training parameters.

        Args:
            game: ChessGame instance.
            nnet: ChessNeuralNet instance.
            args: Configuration settings.
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # Copy of the previous best model
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # Stores training data from recent iterations
        self.skipFirstSelfPlay = False  # Used when loading previous training data

    def executeEpisode(self):
        """
        Plays a single game of self-play, generating training data.

        Returns:
            A list of training examples in the form (canonicalBoard, policy, value).
        """
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # Use MCTS to get move probabilities
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            # Store data for training
            trainExamples.append((canonicalBoard, pi, None))

            # Choose action based on MCTS probabilities
            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            # Check if the game has ended
            gameResult = self.game.getGameEnded(board, curPlayer)
            if gameResult != 0:
                # Assign value to all training examples from this episode
                return [(x[0], x[1], gameResult * ((-1) ** (curPlayer != 1))) for x in trainExamples]

    def learn(self):
        """
        Runs multiple iterations of self-play and training.
        """
        for iteration in range(1, self.args.numIters + 1):
            log.info(f'Starting Iteration #{iteration}...')

            if not self.skipFirstSelfPlay or iteration > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                # Generate training examples through self-play
                for _ in tqdm(range(self.args.numEps), desc="Self-Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # Reset MCTS
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            # Keep only the most recent training examples
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning("Removing oldest training examples.")
                self.trainExamplesHistory.pop(0)

            self.saveTrainExamples(iteration - 1)

            # Prepare training data
            trainExamples = []
            for examples in self.trainExamplesHistory:
                trainExamples.extend(examples)
            shuffle(trainExamples)

            # Train the new model
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.trainNetwork(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # Evaluate new model against the previous best model
            log.info('Evaluating New Model...')
            wins, losses, draws = self.evaluateModels(pmcts, nmcts)

            log.info(f'New Model Wins: {wins}, Previous Model Wins: {losses}, Draws: {draws}')
            if wins + losses == 0 or wins / (wins + losses) < self.args.updateThreshold:
                log.info('Rejecting New Model...')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('Accepting New Model...')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(iteration))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def evaluateModels(self, pmcts, nmcts):
        """
        Plays a set number of games between the previous and new models.

        Returns:
            (wins, losses, draws): Number of games won, lost, and drawn by the new model.
        """
        from Arena import Arena  # Importing here to avoid circular import issues
        from ChessGame import ChessGame

        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game, display=ChessGame.display)
        return arena.playGames(self.args.arenaCompare)

    def getCheckpointFile(self, iteration):
        """
        Returns the filename for a checkpoint.

        Args:
            iteration: Current training iteration.

        Returns:
            A string with the checkpoint filename.
        """
        return f'checkpoint_{iteration}.pth.tar'

    def saveTrainExamples(self, iteration):
        """
        Saves training examples to a file.

        Args:
            iteration: Current training iteration.
        """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        """
        Loads training examples from a file.
        """
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"

        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with training examples not found!')
            if input("Continue? [y/n]") != "y":
                sys.exit()
        else:
            log.info("Loading training examples...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info("Training examples loaded.")
            self.skipFirstSelfPlay = True
