#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:04:16 2023

@author: costantino_ai
"""
import math
import numpy as np
import torchvision

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import PIL

def extract_model_seed_from_filename(file_name):
    """
    Extract model and seed information from the file name.

    This function assumes a specific file name format: 'activations_model-{model_id}_seed-{seed}.pkl'.
    It extracts the 'model_id' and 'seed' information from the file name.

    :param file_name: The file name to extract information from.
    :type file_name: str
    :return: A tuple containing the 'model_id' and 'seed' extracted from the file name.
    :rtype: Tuple[int, int]

    :Example:

    >>> file_name = 'activations_model-123_seed-456.pkl'
    >>> model_id, seed = extract_model_seed_from_filename(file_name)
    >>> print(model_id, seed)
    123 456
    """
    parts = file_name.split('_')
    model_id = parts[1].split('-')[1]
    seed = parts[2].split('-')[1].replace('.pkl', '')
    return model_id, seed

def env_check():
    """
    Check and display environment information for the current PyTorch setup.

    This function prints information about the PyTorch version, Torchvision version, the number of available CUDA devices,
    Pillow version, and the selected device along with the number of available GPUs.

    :return: None
    :rtype: None

    :Example:

    >>> env_check()
    PyTorch Version: 1.9.0
    Torchvision Version: 0.10.0
    CUDA device(s): 1
    Pillow version: 8.2.0
    Using device: cuda:0, Number of GPUs: 1
    """

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print("CUDA device(s): ", torch.cuda.device_count())
    print("Pillow version: ", PIL.__version__)
    device, n_gpus = get_device_info() 
    print(f"Using device: {device}, Number of GPUs: {n_gpus}")
    
def get_device_info():
    """
    Determine the appropriate device for computation and the number of available GPUs.

    This function determines the appropriate PyTorch device for computation and checks the number of available GPUs.
    It returns a tuple containing the selected device as a torch.device object and the number of available GPUs as an integer.

    :return: A tuple containing the selected device and the number of available GPUs.
    :rtype: tuple
    :rtype[0]: torch.device
    :rtype[1]: int

    :Example:

    >>> device, num_gpus = get_device_info()
    >>> print(device, num_gpus)
    cuda:0 1
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        num_gpus = 0  # MPS does not use multiple GPUs
    else:
        device = torch.device("cpu")
        num_gpus = 0

    return device, num_gpus

def initialize_weights(model):
    """
    Initialize the weights of a PyTorch model.

    This function applies different weight initialization strategies based on the type of the layer:
    - For nn.Conv2d layers: Kaiming uniform initialization.
    - For nn.BatchNorm2d layers: Constant initialization with ones for weights and zeros for biases.
    - For nn.Linear layers: Xavier uniform initialization for weights and zeros for biases.

    :param model: The PyTorch model whose weights need to be initialized.
    :type model: torch.nn.Module
    :return: None
    :rtype: None

    :Example:

    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3),
    ...     nn.BatchNorm2d(64),
    ...     nn.Linear(128, 10)
    ... )
    >>> initialize_weights(model)
    """

    def init_conv2d(m):
        """Apply Kaiming uniform initialization to Conv2d layers."""
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    def init_batchnorm2d(m):
        """Initialize BatchNorm2d layers with constant values."""
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

    def init_linear(m):
        """Apply Xavier uniform initialization to Linear layers."""
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

    # Iterate through all modules and apply initialization based on the type
    for m in model.modules():
        try:
            if isinstance(m, nn.Conv2d):
                init_conv2d(m)
            elif isinstance(m, nn.BatchNorm2d):
                init_batchnorm2d(m)
            elif isinstance(m, nn.Linear):
                init_linear(m)
        except Exception as e:
            print(f"Warning: Failed to initialize weights for {m.__class__.__name__}: {e}")


def get_last_level_layer_names(model):
    """
    Extract the names of all last-level layers in a PyTorch neural network.

    This function takes a PyTorch model as input and extracts the names of all last-level layers in the model.
    The last-level layers are typically those directly connected to the model's final output or decision-making.

    :param model: The PyTorch model.
    :type model: torch.nn.Module
    :return: A list containing the names of all last-level layers in the model.
    :rtype: list

    :Example:

    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3),
    ...     nn.BatchNorm2d(64),
    ...     nn.Linear(128, 10)
    ... )
    >>> last_level_layers = get_last_level_layer_names(model)
    >>> print(last_level_layers)
    ['2']
    """
    last_level_layers = []
    for name, module in model.named_modules():
        # Check if the module is a leaf module (no children)
        if not list(module.children()):
            # Exclude the top-level module (the model itself) which is always a leaf
            if name:
                last_level_layers.append(name)

    return last_level_layers

def plot_filters(model, layer_name):
    """
    Plot the filters of a specified convolutional layer in a PyTorch model in a square grid.

    This function takes a PyTorch model and the name of a specific convolutional layer and plots the filters from that layer
    in a square grid for visualization purposes.

    :param model: The PyTorch model.
    :type model: torch.nn.Module
    :param layer_name: The name of the convolutional layer to plot the filters from.
    :type layer_name: str
    :raises ValueError: If the specified layer is not found or is not a convolutional layer.
    :return: None
    :rtype: None

    :Example:

    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3),
    ...     nn.BatchNorm2d(64),
    ...     nn.Linear(128, 10)
    ... )
    >>> plot_filters(model, '0')
    """
    # Extracting the specified layer
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None or not isinstance(layer, torch.nn.modules.conv.Conv2d):
        raise ValueError("Specified layer not found or is not a Conv2D layer.")

    # Extracting the filters from the layer
    filters = layer.weight.data
    if filters.ndim != 4:
        raise ValueError("Layer's weight must be a 4D tensor.")

    num_kernels = filters.size(0)

    # Calculate the grid size (as square as possible)
    num_cols = int(math.ceil(math.sqrt(num_kernels)))
    num_rows = int(math.ceil(num_kernels / num_cols))

    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2))  # Adjust figure size as needed

    for i in range(num_kernels):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        kernel = filters[i].cpu().detach().numpy()
        kernel = (kernel - kernel.mean()) / kernel.std()
        kernel = np.clip(kernel, 0, 1)
        kernel = np.transpose(kernel, (1, 2, 0))

        ax.imshow(kernel, interpolation="nearest")
        ax.axis("off")
        ax.set_title(f"Filter {i}")

    plt.tight_layout()
    plt.show()
    
def plot_layer_activations(activations, num_cols=6):
    """
    Plot the activations of a convolutional layer, automatically handling tensor conversion and detachment.

    This function takes a numpy array of activations from a convolutional layer and plots them.
    It also handles tensor conversion and detachment if the activations are in tensor format.

    :param activations: The activations to be plotted.
    :type activations: numpy.ndarray
    :param num_cols: Number of columns in the plot grid.
    :type num_cols: int
    :return: None
    :rtype: None

    :Example:

    >>> import numpy as np
    >>> activations = np.random.rand(32, 64, 8, 8)  # Example activations
    >>> plot_layer_activations(activations, num_cols=8)
    """
    num_activations = activations.shape[0]
    num_rows = (num_activations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))

    for i, ax in enumerate(axes.flat):
        if i < num_activations:
            ax.imshow(activations[i], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Filter {i}')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()      

def print_model_summary(model, input_shape=(1, 3, 224, 224)):
    """
    Print a detailed summary of a PyTorch model.

    The summary includes layer indices, names, types, input/output shapes, number of parameters, and trainable status.
    The function attempts a forward pass to log input/output shapes and catches any exceptions during this process,
    alerting the user.

    :param model: The PyTorch model to summarize.
    :type model: torch.nn.Module
    :param input_shape: The expected input shape(s) of the model.
                        The first dimension of each shape is considered as the batch size.
                        If a single tuple is provided, a single input will be used.
                        If a list of tuples is provided, multiple inputs will be used.
    :type input_shape: tuple or list of tuples
    :return: None
    :rtype: None

    :Example:

    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3),
    ...     nn.BatchNorm2d(64),
    ...     nn.Linear(128, 10)
    ... )
    >>> input_shape = (1, 3, 224, 224)
    >>> print_model_summary(model, input_shape)
    """

    # Check if input_shape is a list of input shapes or a single input shape
    if isinstance(input_shape, tuple):
        input_shapes = [input_shape]  # Make it a list for uniform processing later
    elif isinstance(input_shape, list):
        input_shapes = input_shape
    else:
        raise TypeError("input_shape must be a tuple or a list of tuples")

    # Check for batch dimension in input shapes and add it if missing
    for i, shape in enumerate(input_shapes):
        if len(shape) == 3:
            print(f"Warning: Adding batch dimension to input shape {shape}")
            input_shapes[i] = (1,) + shape  # Add batch dimension

    # Formatting and printing the summary
    def _format_and_print_summary():
        # Function to format and print the model summary
        header_format = " | ".join(["{:<" + str(max_lengths[heading] + 2) + "}" for heading in headings])
        print("Model Summary:")
        print(header_format.format(*headings))
        print("-" * (sum(max_lengths.values()) + len(max_lengths) * 3))

        for layer_info in layers_info:
            print(header_format.format(*layer_info))

        print("-" * (sum(max_lengths.values()) + len(max_lengths) * 3))
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")

    def _forward_pass(model, input_shapes):
        # Function to perform a forward pass and record input/output shapes
        shapes = {}
        hooks = []

        def register_hook(module):
            # Register a hook to capture input/output shapes
            def hook(module, input, output):
                if hasattr(output, 'shape'):
                    input_shape = str(input[0].shape)[11:-1]  # Format input shape
                    output_shape = str(output.shape)[11:-1]  # Format output shape
                    shapes[module] = (input_shape, output_shape)
            if not isinstance(module, nn.Sequential) and \
               not isinstance(module, nn.ModuleList) and \
               not (module == model):
                hooks.append(module.register_forward_hook(hook))

        model.apply(register_hook)

        # Creating dummy inputs for the forward pass
        dummy_inputs = [torch.rand(*shape) for shape in input_shapes]
        
        # Move inputs to the device of the model
        device = next(model.parameters()).device
        dummy_inputs = [inp.to(device) for inp in dummy_inputs]

        try:
            with torch.no_grad():
                model(*dummy_inputs)  # Forward pass with dummy inputs
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")

        for hook in hooks:
            hook.remove()  # Remove hooks after forward pass

        return shapes

    # Handling DataParallel wrapper
    model = model.module if isinstance(model, nn.DataParallel) else model

    # Perform a forward pass to get input-output shapes
    layer_shapes = _forward_pass(model, input_shapes)

    # Header information for summary table
    headings = ["Index", "Name", "Layer Type", "Input Shape", "Output Shape", "Param #", "Trainable"]
    max_lengths = {heading: len(heading) for heading in headings}

    # Collecting layer details for summary
    layers_info = []
    total_params, trainable_params = 0, 0
    for index, (name, layer) in enumerate(model.named_modules()):
        if name == "":
            continue  # Skip the top-level module itself

        layer_type = layer.__class__.__name__
        num_params = sum(p.numel() for p in layer.parameters())
        is_trainable = any(p.requires_grad for p in layer.parameters())
        in_shape, out_shape = layer_shapes.get(layer, ('-', '-'))

        num_params_str = '-' if num_params == 0 else str(num_params)
        trainable_str = '-' if num_params == 0 else str(is_trainable)

        layer_info = [str(index), name, layer_type, in_shape, out_shape, num_params_str, trainable_str]
        layers_info.append(layer_info)

        # Update max lengths for formatting
        for i, info in enumerate(layer_info):
            max_lengths[headings[i]] = max(max_lengths[headings[i]], len(info))

        total_params += num_params
        trainable_params += num_params if is_trainable else 0

    _format_and_print_summary()  # Print the formatted summary

def freeze_layers(model, index_to=None):
    """
    Freeze layers in a PyTorch model for fine-tuning.

    This function takes a PyTorch model and an optional index 'index_to' to freeze layers.
    If 'index_to' is provided, it freezes layers up to the specified index.
    A negative index means to freeze all layers except the last 'index_to' layers.

    :param model: The PyTorch model to freeze the layers of.
    :type model: torch.nn.Module
    :param index_to: The index of the last layer to freeze (optional).
                    If not provided, all layers will be frozen.
    :type index_to: int, optional
    :return: The modified PyTorch model with frozen layers.
    :rtype: torch.nn.Module

    :Example:

    >>> import torch.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3),
    ...     nn.BatchNorm2d(64),
    ...     nn.Linear(128, 10)
    ... )
    >>> frozen_model = freeze_layers(model, index_to=1)
    """
    # Check if the model is wrapped in DataParallel
    is_parallel = isinstance(model, nn.DataParallel)

    if is_parallel:
        model = model.module

    # Get a list of all layers with more than 0 parameters
    layers = [
        module
        for module in model.modules()
        if isinstance(module, nn.Module)
        and len(list(module.parameters())) > 0
        and not isinstance(module, nn.Sequential)
    ]

    # Handle negative index_to values
    if index_to is not None and index_to < 0:
        index_to = len(layers) + index_to

    # Set the default value for index_to if not provided
    if index_to is None:
        index_to = len(layers)

    # Freeze layers up to the specified index
    for idx, layer in enumerate(layers):
        if idx < index_to:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    if is_parallel:
        # Return the modified model in DataParallel format
        model = nn.DataParallel(model)

    return model
