import logging
import os
import coloredlogs
from ChessCoach import ChessCoach
from ChessGame import ChessGame
from ChessNeuralNet import ChessNeuralNet
from utils import dotdict

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change to DEBUG for more detailed logs.

# Training Parameters
args = dotdict({
    'numIters': 3,              # Total training iterations 1000
    'numEps': 6,                 # Self-play games per iteration 100
    'tempThreshold': 15,           # Turns before MCTS temp goes to 0 (reduces randomness)
    'updateThreshold': 0.6,        # % of games new model must win to replace old model
    'maxlenOfQueue': 200000,       # Training history size
    'numMCTSSims': 25,             # Number of MCTS simulations per move
    'arenaCompare': 40,            # Number of evaluation games between old/new model
    'cpuct': 1,                    # UCT exploration parameter

    'checkpoint': './models/',     # Folder to save models
    'load_model': False,           # Load a pre-trained model?
    'load_folder_file': ('./models', 'best.pth.tar'),  # Model to load
    'numItersForTrainExamplesHistory': 20,  # How many past iterations of training data to keep
})

def main():
    """
    Main function to start AlphaZero Chess training.
    """
    print("Starting AlphaZero Chess...")
    log.info('Initializing ChessGame...')
    game = ChessGame()

    print("ChessGame initialized.")
    log.info('Initializing Neural Network...')
    nnet = ChessNeuralNet(game)

    print("Neural Network initialized.")
    if args.load_model:
        log.info(f'Loading model from {args.load_folder_file[0]}/{args.load_folder_file[1]}...')
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Starting from scratch (no pre-trained model loaded)!')

    print("Model loaded.")
    log.info('Initializing ChessCoach...')
    coach = ChessCoach(game, nnet, args)

    if args.load_model:
        log.info("Loading training examples from file...")
        coach.loadTrainExamples()

    print("Training examples loaded.")
    log.info('Starting the training process! 🎉')
    coach.learn()
    print("Training complete.")

if __name__ == "__main__":
    main()
