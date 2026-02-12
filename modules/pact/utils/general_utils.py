"""
General utility functions for the TRELLIS project.

This module provides a collection of utility functions organized into several categories:
- Dictionary utilities: Operations for merging, traversal, reduction, and flattening of dictionaries
- Context management: Tools for managing nested context managers
- Image utilities: Functions for creating image grids and annotating images
- Debug utilities: Tools for numerical comparison and tolerance checking
- Print utilities: Helpers for text formatting and display
"""

import re
import numpy as np
import cv2
import torch
import contextlib


# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries recursively with conflict detection.
    
    Args:
        dicta: First dictionary to merge
        dictb: Second dictionary to merge
        prefix: Used for error reporting to track nested keys
        
    Returns:
        A new merged dictionary
        
    Raises:
        ValueError: If the same key exists in both dictionaries but has different types
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    # Get all unique keys from both dictionaries
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            # If key exists in both, recursively merge if both are dictionaries
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                # Raise error for conflicting non-dictionary values
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            # Copy values from first dictionary
            dict_[key] = dicta[key]
        else:
            # Copy values from second dictionary
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    
    This is the public interface that wraps _dict_merge with default prefix.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    
    Args:
        dic: Dictionary to process
        func: Default function to apply to each leaf value
        special_func: Dictionary mapping keys to special functions for specific keys
        
    Returns:
        Transformed dictionary with function applied to all leaf values
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            # Recursively process nested dictionaries
            dic[key] = dict_foreach(dic[key], func)
        else:
            # Apply special function if key is in special_func, otherwise use default
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    
    Args:
        dicts: List of dictionaries to reduce
        func: Default reduction function (takes a list of values, returns single value)
        special_func: Dictionary mapping keys to special reduction functions
        
    Returns:
        A single merged dictionary with values reduced according to the provided functions
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    # Collect all unique keys across all dictionaries
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        # Extract values for this key from all dictionaries
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            # Recursively reduce nested dictionaries
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            # Apply special function if key is in special_func, otherwise use default
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Check if any value in the dictionary satisfies the given predicate function.
    
    Args:
        dic: Dictionary to check
        func: Predicate function that returns True/False for each leaf value
        
    Returns:
        True if any leaf value satisfies the predicate, False otherwise
    
    dict any time: {'step': 16.795613527297974, 'elapsed': 16.795613527297974}
    dict any step: 16.795613527297974
    dict any elapsed: 16.795613527297974
    dict any loss: {'bin_3': {'mse': nan}, 'bin_5': {'mse': nan}, 'mse': nan, 'loss': nan, 'bin_4': {'mse': nan}, 'bin_7': {'mse': nan}, 'bin_8': {'mse': nan}}
    dict any bin_3: {'mse': nan}
    dict any mse: nan
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        # print(f"dict any {key}: {dic[key]}")
        if isinstance(dic[key], dict):
            # Recursively check nested dictionaries
            if dict_any(dic[key], func):
                return True
        else:
            # Check current value against predicate
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Check if all values in the dictionary satisfy the given predicate function.
    
    Args:
        dic: Dictionary to check
        func: Predicate function that returns True/False for each leaf value
        
    Returns:
        True if all leaf values satisfy the predicate, False otherwise
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            # Recursively check nested dictionaries
            if not dict_all(dic[key], func):
                return False
        else:
            # Check current value against predicate
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    
    Args:
        dic: Dictionary to flatten
        sep: Separator string used to join key levels in the flattened dictionary
        
    Returns:
        A flattened dictionary with compound keys joined by the separator
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            # Recursively flatten nested dictionaries and prefix with current key
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            # Copy leaf values directly
            flat_dict[key] = dic[key]
    return flat_dict


# Context utils
@contextlib.contextmanager
def nested_contexts(*contexts):
    """
    Create a single context manager from multiple context manager factories.
    
    This utility allows combining multiple context managers into a single
    context manager, simplifying code that needs to use multiple contexts.
    
    Args:
        *contexts: Context manager factory functions (functions that return context managers)
        
    Yields:
        Control to the inner block when all contexts are entered
    """
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield


# Image utils
def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    """
    Arrange multiple images into a grid.
    
    Args:
        images: List of images to arrange
        nrow: Number of rows (calculated if not provided)
        ncol: Number of columns (calculated if not provided)
        aspect_ratio: Desired width/height ratio for the grid layout
        
    Returns:
        A single image containing the grid of input images
    """
    num_images = len(images)
    # Calculate grid dimensions if not explicitly provided
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            # Calculate rows to achieve desired aspect ratio
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            # Default to a roughly square grid
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        # Calculate rows based on fixed columns
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        # Calculate columns based on fixed rows
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
    
    # Create empty grid with appropriate dimensions
    if images[0].ndim == 2:
        # Grayscale images
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype)
    else:
        # Color images
        grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    
    # Place each image in the grid
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    """
    Add text notes to an image by padding the bottom and adding text.
    
    Args:
        img: Input image
        notes: Text to add at the bottom of the image
        
    Returns:
        Image with notes added
    """
    # Add padding at the bottom for the notes
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        # Add text to the padded area
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with optional text notes at the bottom.
    
    Args:
        img: Input image (numpy array or PyTorch tensor)
        path: File path to save the image
        notes: Optional text to add at the bottom of the image
    """
    # Convert PyTorch tensor to numpy if needed
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    # Scale floating point images to 0-255 range
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # Add notes to the image
    img = notes_on_image(img, notes)
    # Save with proper color conversion for OpenCV
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# debug utils
def atol(x, y):
    """
    Absolute tolerance - computes absolute difference between x and y.
    
    Useful for numerical comparisons when absolute error matters.
    
    Args:
        x, y: Tensors to compare
        
    Returns:
        Absolute difference |x - y|
    """
    return torch.abs(x - y)


def rtol(x, y):
    """
    Relative tolerance - computes relative difference between x and y.
    
    Useful for numerical comparisons when relative error matters,
    especially when comparing values of different magnitudes.
    
    Args:
        x, y: Tensors to compare
        
    Returns:
        Relative difference |x - y| / max(|x|, |y|)
    """
    return torch.abs(x - y) / torch.clamp_min(torch.maximum(torch.abs(x), torch.abs(y)), 1e-12)


# print utils
def indent(s, n=4):
    """
    Indent a multi-line string.
    
    Args:
        s: Input string to indent
        n: Number of spaces to add before each line (except the first)
        
    Returns:
        Indented string with all lines except the first indented by n spaces
    """
    lines = s.split('\n')
    for i in range(1, len(lines)):
        lines[i] = ' ' * n + lines[i]
    return '\n'.join(lines)
