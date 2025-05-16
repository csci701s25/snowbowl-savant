import torch
from torch.utils.data import DataLoader
from typing import Type
from torch import nn

def train_model(model_class: Type[nn.Module], input_dim: int, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 25, 
                lr: float = 0.002, verbose: bool = True) -> None:
    """
    Trains a binary classifier using BCE loss with an Adam optimizer.

    Args:
        - model_class: A torch.nn.Module class
        - input_dim: Input feature dimension
        - train_loader: DataLoader for training data
        - test_loader: DataLoader for testing data
        - epochs: Number of epochs to train
        - lr: Learning rate for optimizer
        - verbose: Whether to print progress
    
    Returns:
        Trained model
    """
    model = model_class(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_y.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        avg_train_loss = total_train_loss / total_train_samples

        model.eval()
        total_test_loss = 0.0
        total_test_samples = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                batch_size = batch_y.size(0)
                total_test_loss += loss.item() * batch_size
                total_test_samples += batch_size

                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_size

        avg_test_loss = total_test_loss / total_test_samples
        accuracy = 100 * correct / total

        if verbose:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, Accuracy = {accuracy:.2f}%")

    return model
