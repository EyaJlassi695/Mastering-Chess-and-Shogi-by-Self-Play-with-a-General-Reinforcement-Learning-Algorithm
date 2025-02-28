import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from ChessGame import ChessGame

class ChessNeuralNet(nn.Module):
    """
    A simple convolutional neural network for AlphaZero Chess.
    """
    def __init__(self, game, lr=0.001):
        super(ChessNeuralNet, self).__init__()
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.lr = lr

        # Convolutional layers to extract features from the board
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        
        # Fully connected layers for policy head
        self.policy_fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.policy_fc2 = nn.Linear(1024, self.action_size)

        # Fully connected layers for value head
        self.value_fc1 = nn.Linear(128 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        # Optimizer and loss functions
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass through the network.
        
        Input:
            x: Board state tensor of shape (batch, 6, 8, 8).
        
        Returns:
            policy: Probability distribution over possible moves.
            value: Estimated game outcome [-1, 1].
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        x = x.view(x.size(0), -1)  # Flatten feature maps

        # Policy Head
        policy = self.relu(self.policy_fc1(x))
        policy = self.policy_fc2(policy)
        policy = self.softmax(policy)

        # Value Head
        value = self.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        value = self.tanh(value)

        return policy, value

    def trainNetwork(self, examples):
        """
        Train the neural network with self-play examples.

        Input:
            examples: List of (board, policy, value) tuples.
        """
        self.train()
        batch_boards, batch_policies, batch_values = zip(*examples)

        # Convert all boards to matrix representations before tensor conversion
        batch_boards = [ChessGame().boardToMatrix(board) if not isinstance(board, np.ndarray) else board for board in batch_boards]

        # Convert to PyTorch tensors
        boards = torch.tensor(np.array(batch_boards), dtype=torch.float32)
        policies = torch.tensor(np.array(batch_policies), dtype=torch.float32)
        values = torch.tensor(np.array(batch_values), dtype=torch.float32).unsqueeze(1)

        # Forward pass
        self.optimizer.zero_grad()
        pred_policies, pred_values = self.forward(boards)

        # Compute loss
        policy_loss = self.criterion_policy(pred_policies, policies)
        value_loss = self.criterion_value(pred_values, values)
        total_loss = policy_loss + value_loss

        # Backpropagation
        total_loss.backward()
        self.optimizer.step()


    def predict(self, board):
        """
        Runs the neural network prediction on a given board state.

        Args:
            board (chess.Board): The board to predict for.

        Returns:
            pi (np.ndarray): Policy vector (move probabilities).
            v (float): Value estimate (-1 to 1).
        """
        game=ChessGame()
        board_matrix = game.boardToMatrix(board)  # Convert chess.Board() to 8x8 matrix
        board_tensor = torch.tensor(board_matrix, dtype=torch.float32).unsqueeze(0)  # Convert to tensor

        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations for inference
            pi, v = self.forward(board_tensor)
            pi = pi.numpy().flatten()

        #print(f"Predicted policy vector: {pi}")  # Debugging line
        #print(f"Sum of probabilities: {np.sum(pi)}")

        return pi, v.item()

        return pi.numpy().flatten(), v.item()

    def save_checkpoint(self, folder, filename):
        """
        Save the model weights, creating the folder if necessary.
        """
        if not os.path.exists(folder):  # Check if folder exists
            os.makedirs(folder)  # Create it if it doesn't

        filepath = os.path.join(folder, filename)  # Full path
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, folder, filename):
        """
        Load the model weights.
        """
        filepath = f"{folder}/{filename}"
        self.load_state_dict(torch.load(filepath))
