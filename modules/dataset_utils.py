#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:42:46 2023

@author: costantino_ai
"""
import os
import torch
import chess
from modules import logging
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from modules.models.alphazero.encoder import encodePositionForInference

def get_fmri_dataloader_from_csv(csv_file_path, batch_size=32, shuffle=True, num_workers=8, pin_memory=True):
    """
    Create a DataLoader for the fMRI dataset from a CSV file.

    This function creates a DataLoader for the fMRI dataset based on the information provided in a CSV file.
    The DataLoader allows for efficient batched loading of the dataset, with options for shuffling, specifying
    the batch size, and utilizing multiple workers for data loading.

    :param csv_file_path: Path to the CSV file containing the dataset.
    :type csv_file_path: str
    :param batch_size: How many samples per batch to load. If None, the batch size is set to the size of the dataset.
    :type batch_size: int, optional
    :param shuffle: Whether to shuffle the dataset. Default is True.
    :type shuffle: bool, optional
    :param num_workers: How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. Default is 0.
    :type num_workers: int, optional
    :param pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them. This is only relevant for CUDA-enabled GPUs. Default is False.
    :type pin_memory: bool, optional
    :return: A DataLoader object for the fMRI dataset.
    :rtype: DataLoader

    :Example:

    >>> dataloader = get_fmri_dataloader_from_csv("fmri_dataset.csv", batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    >>> for batch in dataloader:
    ...     # Process the batch
    """
    
    # Initialize the dataset
    fmri_dataset = FMRIDataset(board_dir=csv_file_path)

    # Determine batch size
    if batch_size is None:
        batch_size = len(fmri_dataset)

    # Initialize the DataLoader
    fmri_dataloader = DataLoader(fmri_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return fmri_dataloader

class FMRIDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and processing fMRI data.

    This dataset is designed to load and process fMRI data from a CSV file with specific columns.
    Each row in the CSV file represents a single fMRI data point, and the expected columns are:

    - 'fen': A string representing the chess position in Forsyth-Edwards Notation.
    - 'check': A binary or categorical value indicating a specific condition, e.g., check in chess.
    - 'strategy': A numerical or categorical value representing a specific strategy element.
    - 'visual': A numerical or categorical value related to visual aspects.
    - 'stim_id': A unique identifier for each stimulus or data point.
    - 'filename': A string indicating the filename or path related to the data point.

    The 'fen' column is used to generate the chess board state, which is then processed to create
    the 'position' and 'mask' tensors for further analysis.

    :param fmri_data: A DataFrame containing the fMRI data with the specified columns.
    :type fmri_data: pd.DataFrame
    :param board_dir: The directory where the CSV file is located.
    :type board_dir: str

    Attributes:
        fmri_data (pd.DataFrame): A DataFrame containing the fMRI data with the specified columns.
        board_dir (str): The directory where the CSV file is located.
    """
    def __init__(self, board_dir):
        """
        Initialize the dataset by loading the data from a CSV file.

        Args:
            board_dir (str): The directory of the CSV file containing fMRI data.
        """
        # Attempt to load the data and handle any file-related errors
        try:
            self.fmri_data = pd.read_csv(board_dir)
        except Exception as e:
            logging.error(f"Error loading data from {board_dir}: {e}")
            raise

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: The total number of items.
        """
        return len(self.fmri_data)

    def __getitem__(self, idx):
        """
        Return a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the data for a single fMRI scan.
        """
        # Retrieve the row from the DataFrame
        row = self.fmri_data.iloc[idx]

        # Process the chess board position
        fen_str = row['fen']
        board = chess.Board(fen_str)

        # Get position and mask input
        try:
            position, mask = encodePositionForInference(board, color="white" if board.turn else "black")
        except Exception as e:
            logging.error(f"Error processing position and mask for index {idx}: {e}")
            return None

        # Convert position and mask to tensors and unsqueeze
        position_tensor = torch.from_numpy(position)
        mask_tensor = torch.from_numpy(mask)

        # Create the dictionary to be returned
        data_dict = {
            'position': position_tensor,
            'mask': mask_tensor,
            'check': torch.tensor(row['check']),
            'strategy': torch.tensor(row['strategy']),
            'visual': torch.tensor(row['visual']),
            'stim_id': torch.tensor(row['stim_id']),
            'filename': row['filename'],
            'fen': row['fen']
        }

        return data_dict

def load_MNIST_data(batch_size=32, subset_size=None, train=True, mnist_dir="./datasets/mnist"):
    """
    Load MNIST data with appropriate transformations.

    This function loads the MNIST dataset with specified configurations, including batch size, subset size (for quick experiments),
    and whether to load training or test data. It applies appropriate transformations to the dataset to prepare it for machine learning tasks.

    :param batch_size: Batch size for the DataLoader.
    :type batch_size: int
    :param subset_size: Size of a subset to use (for quick experiments).
    :type subset_size: int or None
    :param train: Whether to load training data or test data.
    :type train: bool
    :param mnist_dir: The directory where the MNIST dataset is located.
    :type mnist_dir: str
    :return: DataLoader for the MNIST dataset.
    :rtype: DataLoader

    :Example:

    >>> dataloader = load_MNIST_data(batch_size=32, subset_size=1000, train=True, mnist_dir="./datasets/mnist")
    >>> for batch in dataloader:
    ...     # Process the batch
    """
    # Base transformations: Grayscale conversion, resizing, and tensor conversion
    base_transform = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]

    # Additional augmentations for training data
    if train:
        augmentation_transforms = [
            transforms.RandomRotation(10),  # Rotations up to 10 degrees
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),  # Slight translations and scaling
        ]
        transform = transforms.Compose(
            augmentation_transforms + base_transform + [transforms.RandomErasing(p=0.2)]
        )  # Random erasing
    else:
        transform = transforms.Compose(base_transform)

    # Load MNIST data
    os.makedirs(mnist_dir, exist_ok=True)
    dataset = datasets.MNIST(root=mnist_dir, train=train, download=True, transform=transform)

    # Subset the dataset if needed
    if subset_size is not None:
        dataset = Subset(dataset, range(subset_size))

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
