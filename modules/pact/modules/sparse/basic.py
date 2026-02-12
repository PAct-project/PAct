"""
Sparse Tensor Implementation for TRELLIS
----------------------------------------

This file implements a unified sparse tensor interface that supports multiple backends (torchsparse and spconv).
Sparse tensors are efficient representations of tensors where most values are zero, storing only non-zero values
and their coordinates. This is particularly useful for 3D point clouds and voxel grids in computer vision and
robotics applications where data is naturally sparse.

The main components of this file are:
- SparseTensor: Core class providing a unified API over different sparse tensor backends
- Utility functions for sparse tensor operations (concatenation, unbinding, broadcasting, etc.)
- Backend-agnostic arithmetic operations for sparse tensors

The implementation abstracts away backend-specific details to allow seamless switching between
torchsparse and spconv while maintaining a consistent interface.
"""

from typing import *
import torch
import torch.nn as nn
from . import BACKEND, DEBUG
SparseTensorData = None # Lazy import


__all__ = [
    'SparseTensor',
    'sparse_batch_broadcast',
    'sparse_batch_op',
    'sparse_cat',
    'sparse_unbind',
]


class SparseTensor:
    """
    Sparse tensor with support for both torchsparse and spconv backends.
    
    Parameters:
    - feats (torch.Tensor): Features of the sparse tensor.
    - coords (torch.Tensor): Coordinates of the sparse tensor.
    - shape (torch.Size): Shape of the sparse tensor.
    - layout (List[slice]): Layout of the sparse tensor for each batch
    - data (SparseTensorData): Sparse tensor data used for convolusion

    NOTE:
    - Data corresponding to a same batch should be contiguous.
    - Coords should be in [0, 1023]
    """
    @overload
    def __init__(self, feats: torch.Tensor, coords: torch.Tensor, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    @overload
    def __init__(self, data, shape: Optional[torch.Size] = None, layout: Optional[List[slice]] = None, **kwargs): ...

    def __init__(self, *args, **kwargs):
        # Lazy import of sparse tensor backend to avoid circular imports and improve startup time
        global SparseTensorData
        if SparseTensorData is None:
            import importlib
            if BACKEND == 'torchsparse':
                SparseTensorData = importlib.import_module('torchsparse').SparseTensor
            elif BACKEND == 'spconv':
                SparseTensorData = importlib.import_module('spconv.pytorch').SparseConvTensor
        
        # print(SparseTensorData)
        # exit(0)
                
        # Determine initialization method based on arguments (method 0: from tensors, method 1: from existing data)
        method_id = 0
        if len(args) != 0:
            method_id = 0 if isinstance(args[0], torch.Tensor) else 1
        else:
            method_id = 1 if 'data' in kwargs else 0

        self.old_index = None  # Placeholder for old indices, if needed

        if method_id == 0:
            # Initialize from feature and coordinate tensors
            feats, coords, shape, layout = args + (None,) * (4 - len(args))
            if 'feats' in kwargs:
                feats = kwargs['feats']
                del kwargs['feats']
            if 'coords' in kwargs:
                coords = kwargs['coords']
                del kwargs['coords']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            if shape is None:
                shape = self.__cal_shape(feats, coords)
            if layout is None:
                layout = self.__cal_layout(coords, shape[0])
            
            # Create backend-specific tensor representation
            if BACKEND == 'torchsparse':
                self.data = SparseTensorData(feats, coords, **kwargs)
            elif BACKEND == 'spconv':
                spatial_shape = list(coords.max(0)[0] + 1)[1:]
                self.data = SparseTensorData(feats.reshape(feats.shape[0], -1), coords, spatial_shape, shape[0], **kwargs)
                self.data._features = feats
        elif method_id == 1:
            # Initialize from existing sparse tensor data
            data, shape, layout = args + (None,) * (3 - len(args))
            if 'data' in kwargs:
                data = kwargs['data']
                del kwargs['data']
            if 'shape' in kwargs:
                shape = kwargs['shape']
                del kwargs['shape']
            if 'layout' in kwargs:
                layout = kwargs['layout']
                del kwargs['layout']

            self.data = data
            if shape is None:
                shape = self.__cal_shape(self.feats, self.coords)
            if layout is None:
                layout = self.__cal_layout(self.coords, shape[0])

        # Store metadata
        self._shape = shape
        self._layout = layout
        self._scale = kwargs.get('scale', (1, 1, 1))
        self._spatial_cache = kwargs.get('spatial_cache', {})

        # Validate tensor properties in debug mode
        if DEBUG:
            try:
                assert self.feats.shape[0] == self.coords.shape[0], f"Invalid feats shape: {self.feats.shape}, coords shape: {self.coords.shape}"
                assert self.shape == self.__cal_shape(self.feats, self.coords), f"Invalid shape: {self.shape}"
                assert self.layout == self.__cal_layout(self.coords, self.shape[0]), f"Invalid layout: {self.layout}"
                for i in range(self.shape[0]):
                    assert torch.all(self.coords[self.layout[i], 0] == i), f"The data of batch {i} is not contiguous"
            except Exception as e:
                print('Debugging information:')
                print(f"- Shape: {self.shape}")
                print(f"- Layout: {self.layout}")
                print(f"- Scale: {self._scale}")
                print(f"- Coords: {self.coords}")
                raise e
        
    def __cal_shape(self, feats, coords):
        """
        Calculate the shape of the sparse tensor from features and coordinates.
        
        This method determines the overall shape of the sparse tensor by examining:
        - The batch dimension (from max coordinate value in first column + 1)
        - The feature dimensions (from the feature tensor shape)
        
        Args:
            feats (torch.Tensor): Feature tensor of shape (N, C1, C2, ...)
            coords (torch.Tensor): Coordinate tensor of shape (N, D+1) where
                                  first column contains batch indices
        
        Returns:
            torch.Size: Shape of the sparse tensor as (batch_size, C1, C2, ...)
        """
        shape = []
        # First dimension is the batch size (max batch index + 1)
        shape.append(coords[:, 0].max().item() + 1)
        # Remaining dimensions match the feature tensor's dimensions
        shape.extend([*feats.shape[1:]])
        return torch.Size(shape)
    
    def __cal_layout(self, coords, batch_size):
        """
        Calculate the layout of each batch in the sparse tensor.
        
        This method computes slice objects to efficiently index into specific batches
        within the coordinate and feature tensors. It assumes that coordinates are
        sorted by batch index (first column).
        
        Algorithm:
        1. Count how many elements belong to each batch using bincount
        2. Calculate cumulative sums to find ending offsets for each batch
        3. Create slice objects representing the range of indices for each batch
        
        Args:
            coords (torch.Tensor): Coordinate tensor with first column as batch indices
            batch_size (int): Number of batches in the sparse tensor
            
        Returns:
            List[slice]: List of slice objects where layout[i] indexes all elements
                         belonging to batch i
        """
        # Count number of points in each batch
        seq_len = torch.bincount(coords[:, 0], minlength=batch_size)
        # Calculate ending position of each batch
        offset = torch.cumsum(seq_len, dim=0) 
        # Create slices for each batch from (end_prev_batch, end_current_batch)
        layout = [slice((offset[i] - seq_len[i]).item(), offset[i].item()) for i in range(batch_size)]
        return layout
    
    @property
    def shape(self) -> torch.Size:
        """Return the shape of the sparse tensor"""
        return self._shape
    
    def dim(self) -> int:
        """Return the number of dimensions of the sparse tensor"""
        return len(self.shape)
    
    @property
    def layout(self) -> List[slice]:
        """Return the layout of each batch in the sparse tensor"""
        return self._layout

    @property
    def feats(self) -> torch.Tensor:
        """Return the features tensor with backend-specific access"""
        if BACKEND == 'torchsparse':
            return self.data.F
        elif BACKEND == 'spconv':
            return self.data.features
    
    @feats.setter
    def feats(self, value: torch.Tensor):
        """Set the features tensor with backend-specific access"""
        if BACKEND == 'torchsparse':
            self.data.F = value
        elif BACKEND == 'spconv':
            self.data.features = value

    @property
    def coords(self) -> torch.Tensor:
        """Return the coordinates tensor with backend-specific access"""
        if BACKEND == 'torchsparse':
            return self.data.C
        elif BACKEND == 'spconv':
            return self.data.indices
        
    @coords.setter
    def coords(self, value: torch.Tensor):
        """Set the coordinates tensor with backend-specific access"""
        if BACKEND == 'torchsparse':
            self.data.C = value
        elif BACKEND == 'spconv':
            self.data.indices = value

    @property
    def dtype(self):
        """Return the data type of the sparse tensor's features"""
        return self.feats.dtype

    @property
    def device(self):
        """Return the device of the sparse tensor's features"""
        return self.feats.device

    @overload
    def to(self, dtype: torch.dtype) -> 'SparseTensor': ...

    @overload
    def to(self, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None) -> 'SparseTensor': ...

    def to(self, *args, **kwargs) -> 'SparseTensor':
        """
        Move the sparse tensor to the specified device and/or change its data type.
        Mimics the PyTorch tensor.to() method.
        """
        device = None
        dtype = None
        if len(args) == 2:
            device, dtype = args
        elif len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype = args[0]
            else:
                device = args[0]
        if 'dtype' in kwargs:
            assert dtype is None, "to() received multiple values for argument 'dtype'"
            dtype = kwargs['dtype']
        if 'device' in kwargs:
            assert device is None, "to() received multiple values for argument 'device'"
            device = kwargs['device']

        # print(self.feats)
        # print(self.coords)
        # print(SparseTensorData)
        new_feats = self.feats.to(device=device, dtype=dtype)
        new_coords = self.coords.to(device=device)
        return self.replace(new_feats, new_coords)

    def type(self, dtype):
        """Convert the sparse tensor to the specified data type"""
        new_feats = self.feats.type(dtype)
        return self.replace(new_feats)

    def cpu(self) -> 'SparseTensor':
        """Move the sparse tensor to CPU memory"""
        new_feats = self.feats.cpu()
        new_coords = self.coords.cpu()
        return self.replace(new_feats, new_coords)
    
    def cuda(self) -> 'SparseTensor':
        """Move the sparse tensor to CUDA memory"""
        new_feats = self.feats.cuda()
        new_coords = self.coords.cuda()
        return self.replace(new_feats, new_coords)

    def half(self) -> 'SparseTensor':
        """Convert the sparse tensor to half precision"""
        new_feats = self.feats.half()
        return self.replace(new_feats)
    
    def float(self) -> 'SparseTensor':
        """Convert the sparse tensor to single precision"""
        new_feats = self.feats.float()
        return self.replace(new_feats)
    
    def detach(self) -> 'SparseTensor':
        """Detach the sparse tensor from the computation graph"""
        new_coords = self.coords.detach()
        new_feats = self.feats.detach()
        return self.replace(new_feats, new_coords)

    def dense(self) -> torch.Tensor:
        """Convert the sparse tensor to a dense tensor representation"""
        if BACKEND == 'torchsparse':
            return self.data.dense()
        elif BACKEND == 'spconv':
            return self.data.dense()

    def reshape(self, *shape) -> 'SparseTensor':
        """Reshape the feature dimensions of the sparse tensor"""
        new_feats = self.feats.reshape(self.feats.shape[0], *shape)
        return self.replace(new_feats)
    
    def unbind(self, dim: int) -> List['SparseTensor']:
        """Unbind the sparse tensor along the specified dimension"""
        return sparse_unbind(self, dim)

    def replace(self, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> 'SparseTensor':
        """
        Create a new sparse tensor with the specified features and optionally new coordinates.
        Preserves other properties like stride, spatial range, and caches.
        """
        new_shape = [self.shape[0]]
        new_shape.extend(feats.shape[1:])
        if BACKEND == 'torchsparse':
            new_data = SparseTensorData(
                feats=feats,
                coords=self.data.coords if coords is None else coords,
                stride=self.data.stride,
                spatial_range=self.data.spatial_range,
            )
            new_data._caches = self.data._caches
        elif BACKEND == 'spconv':
            new_data = SparseTensorData(
                self.data.features.reshape(self.data.features.shape[0], -1),
                self.data.indices,
                self.data.spatial_shape,
                self.data.batch_size,
                self.data.grid,
                self.data.voxel_num,
                self.data.indice_dict
            )
            new_data._features = feats
            new_data.benchmark = self.data.benchmark
            new_data.benchmark_record = self.data.benchmark_record
            new_data.thrust_allocator = self.data.thrust_allocator
            new_data._timer = self.data._timer
            new_data.force_algo = self.data.force_algo
            new_data.int8_scale = self.data.int8_scale
            if coords is not None:
                new_data.indices = coords
        new_tensor = SparseTensor(new_data, shape=torch.Size(new_shape), layout=self.layout, scale=self._scale, spatial_cache=self._spatial_cache)
        return new_tensor

    @staticmethod
    def full(aabb, dim, value, dtype=torch.float32, device=None) -> 'SparseTensor':
        """
        Create a sparse tensor with uniform values within an axis-aligned bounding box.
        
        Args:
            aabb: [x_min, y_min, z_min, x_max, y_max, z_max] defining the bounding box
            dim: (batch_size, feature_dim) tuple defining tensor dimensions
            value: Value to fill the tensor with
            dtype: Data type for features
            device: Device to create the tensor on
        """
        N, C = dim
        x = torch.arange(aabb[0], aabb[3] + 1)
        y = torch.arange(aabb[1], aabb[4] + 1)
        z = torch.arange(aabb[2], aabb[5] + 1)
        coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        coords = torch.cat([
            torch.arange(N).view(-1, 1).repeat(1, coords.shape[0]).view(-1, 1),
            coords.repeat(N, 1),
        ], dim=1).to(dtype=torch.int32, device=device)
        feats = torch.full((coords.shape[0], C), value, dtype=dtype, device=device)
        return SparseTensor(feats=feats, coords=coords)

    def __merge_sparse_cache(self, other: 'SparseTensor') -> dict:
        """Merge the spatial caches of two sparse tensors"""
        new_cache = {}
        for k in set(list(self._spatial_cache.keys()) + list(other._spatial_cache.keys())):
            if k in self._spatial_cache:
                new_cache[k] = self._spatial_cache[k]
            if k in other._spatial_cache:
                if k not in new_cache:
                    new_cache[k] = other._spatial_cache[k]
                else:
                    new_cache[k].update(other._spatial_cache[k])
        return new_cache

    def __neg__(self) -> 'SparseTensor':
        """Negate the sparse tensor's values"""
        return self.replace(-self.feats)
    
    def __elemwise__(self, other: Union[torch.Tensor, 'SparseTensor'], op: callable) -> 'SparseTensor':
        """
        Apply an elementwise operation between this sparse tensor and another tensor.
        Handles broadcasting when necessary.
        """
        if isinstance(other, torch.Tensor):
            try:
                other = torch.broadcast_to(other, self.shape)
                other = sparse_batch_broadcast(self, other)
            except:
                pass
        if isinstance(other, SparseTensor):
            other = other.feats
        new_feats = op(self.feats, other)
        new_tensor = self.replace(new_feats)
        if isinstance(other, SparseTensor):
            new_tensor._spatial_cache = self.__merge_sparse_cache(other)
        return new_tensor

    def __add__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Add a tensor or value to this sparse tensor"""
        return self.__elemwise__(other, torch.add)

    def __radd__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Add this sparse tensor to a tensor or value (reversed)"""
        return self.__elemwise__(other, torch.add)
    
    def __sub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Subtract a tensor or value from this sparse tensor"""
        return self.__elemwise__(other, torch.sub)
    
    def __rsub__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Subtract this sparse tensor from a tensor or value (reversed)"""
        return self.__elemwise__(other, lambda x, y: torch.sub(y, x))

    def __mul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Multiply this sparse tensor by a tensor or value"""
        return self.__elemwise__(other, torch.mul)

    def __rmul__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Multiply a tensor or value by this sparse tensor (reversed)"""
        return self.__elemwise__(other, torch.mul)

    def __truediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Divide this sparse tensor by a tensor or value"""
        return self.__elemwise__(other, torch.div)

    def __rtruediv__(self, other: Union[torch.Tensor, 'SparseTensor', float]) -> 'SparseTensor':
        """Divide a tensor or value by this sparse tensor (reversed)"""
        return self.__elemwise__(other, lambda x, y: torch.div(y, x))

    def __getitem__(self, idx):
        """
        Extract a batch or subset of batches from the sparse tensor.
        Support for integer, slice, and tensor indexing.
        """
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            idx = range(*idx.indices(self.shape[0]))
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
                idx = idx.nonzero().squeeze(1)
            elif idx.dtype in [torch.int32, torch.int64]:
                assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
            else:
                raise ValueError(f"Unknown index type: {idx.dtype}")
        else:
            raise ValueError(f"Unknown index type: {type(idx)}")
        
        coords = []
        feats = []
        old_index_list = []
        for new_idx, old_idx in enumerate(idx):
            coords.append(self.coords[self.layout[old_idx]].clone())
            # print(f"slice: old index{old_idx}, new index: {new_idx}")
            old_index_list.append(old_idx)
            coords[-1][:, 0] = new_idx
            feats.append(self.feats[self.layout[old_idx]])
        coords = torch.cat(coords, dim=0).contiguous()
        feats = torch.cat(feats, dim=0).contiguous()
        self.old_index = old_index_list
        return SparseTensor(feats=feats, coords=coords)
    def clone(self) -> 'SparseTensor':
        """Create a deep copy of the sparse tensor"""
        return self.replace(self.feats.clone(), self.coords.clone())
    
    # def get_item_preserve_batch(self, idx):
    #     """
    #     Extract a batch or subset of batches from the sparse tensor without renumbering batch indices.
    #     Unlike __getitem__, this method preserves the original batch IDs in the coords tensor.
        
    #     Args:
    #         idx: Integer, slice, torch.Tensor, or tuple specifying which batch(es) to extract.
    #              When a tuple is provided, it's used for direct slicing of the underlying data.
                
    #     Returns:
    #         SparseTensor: A new sparse tensor with the selected batches and original batch IDs
    #     """
    #     if isinstance(idx, tuple):
    #         # Direct slice-based indexing
    #         coords_slice = self.coords[idx]
    #         feats_slice = self.feats[idx]
    #         return SparseTensor(feats=feats_slice, coords=coords_slice)
        
    #     if isinstance(idx, int):
    #         idx = [idx]
    #     elif isinstance(idx, slice):
    #         idx = range(*idx.indices(self.shape[0]))
    #     elif isinstance(idx, torch.Tensor):
    #         if idx.dtype == torch.bool:
    #             assert idx.shape == (self.shape[0],), f"Invalid index shape: {idx.shape}"
    #             idx = idx.nonzero().squeeze(1)
    #         elif idx.dtype in [torch.int32, torch.int64]:
    #             assert len(idx.shape) == 1, f"Invalid index shape: {idx.shape}"
    #         else:
    #             raise ValueError(f"Unknown index type: {idx.dtype}")
    #     else:
    #         raise ValueError(f"Unknown index type: {type(idx)}")
        
    #     coords = []
    #     feats = []
    #     for old_idx in idx:
    #         coords.append(self.coords[self.layout[old_idx]].clone())
    #         # Keep original batch ID (don't modify coords[:, 0])
    #         feats.append(self.feats[self.layout[old_idx]])
        
    #     coords = torch.cat(coords, dim=0).contiguous()
    #     feats = torch.cat(feats, dim=0).contiguous()
        
    #     # Create new SparseTensor with preserved batch IDs
    #     return SparseTensor(feats=feats, coords=coords)
    

    def register_spatial_cache(self, key, value) -> None:
        """
        Register a spatial cache.
        The spatial cache can be any thing you want to cache.
        The registery and retrieval of the cache is based on current scale.
        """
        scale_key = str(self._scale)
        if scale_key not in self._spatial_cache:
            self._spatial_cache[scale_key] = {}
        self._spatial_cache[scale_key][key] = value

    def get_spatial_cache(self, key=None):
        """
        Get a spatial cache.
        If key is None, return all caches for the current scale.
        Otherwise, return the cache associated with the specified key.
        """
        scale_key = str(self._scale)
        cur_scale_cache = self._spatial_cache.get(scale_key, {})
        if key is None:
            return cur_scale_cache
        return cur_scale_cache.get(key, None)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a tensor to a sparse tensor along the batch dimension.
    
    Args:
        input (SparseTensor): Sparse tensor to broadcast to
        other (torch.Tensor): Tensor to broadcast
        
    Returns:
        torch.Tensor: Broadcasted tensor matching the sparse tensor's layout
    """
    coords, feats = input.coords, input.feats
    broadcasted = torch.zeros_like(feats)
    for k in range(input.shape[0]):
        broadcasted[input.layout[k]] = other[k]
    return broadcasted


def sparse_batch_op(input: SparseTensor, other: torch.Tensor, op: callable = torch.add) -> SparseTensor:
    """
    Broadcast a 1D tensor to a sparse tensor along the batch dimension then perform an operation.
    
    Args:
        input (SparseTensor): Sparse tensor to operate on
        other (torch.Tensor): 1D tensor to broadcast
        op (callable): Operation to perform after broadcasting. Defaults to torch.add.
        
    Returns:
        SparseTensor: Result of the operation
    """
    return input.replace(op(input.feats, sparse_batch_broadcast(input, other)))

def sparse_cat(inputs: List[SparseTensor], dim: int = 0) -> SparseTensor:
    """
    Concatenate a list of sparse tensors along a specified dimension.
    
    This function handles two types of concatenation:
    1. Batch concatenation (dim=0): Combines multiple sparse tensors by stacking their batches,
       adjusting batch indices to maintain proper batch ordering.
    2. Feature concatenation (dim>0): Combines features while maintaining the same coordinate structure,
       useful for concatenating different feature channels for the same spatial locations.
    
    Args:
        inputs (List[SparseTensor]): List of sparse tensors to concatenate. All tensors must have
                                    compatible shapes for the requested concatenation dimension.
        dim (int): Dimension along which to concatenate.
                   - If 0, batches are concatenated (increasing batch indices)
                   - If >0, features are concatenated (same coordinates, more features)
                   
    Returns:
        SparseTensor: A new sparse tensor with concatenated data
    """
    if dim == 0:
        # Concatenate batches - requires adjusting batch indices in coordinates
        start = 0
        coords = []
        
        # Process each input sparse tensor
        for input in inputs:
            # Create a copy of coordinates to avoid modifying the original
            current_coords = input.coords.clone()

            # print("current coords", current_coords[:, 0])
            
            # Adjust batch indices (first column of coordinates) to maintain proper batch ordering
            # Each tensor's batch indices are offset by the sum of previous tensors' batch sizes
            current_coords[:, 0] += start
            
            # print("current coords", current_coords[:, 0])
            # Add to coordinate list and update the batch counter
            coords.append(current_coords)

            # print("shape of input", input.shape)

            start += input.shape[0]
            
            # print("start number", start)
        
        # Concatenate all adjusted coordinates into a single tensor
        coords = torch.cat(coords, dim=0)
        
        # Concatenate feature values in the same order as coordinates
        feats = torch.cat([input.feats for input in inputs], dim=0)
        
        # Create a new sparse tensor with combined coordinates and features
        output = SparseTensor(
            coords=coords,
            feats=feats,
        )
    else:
        # Concatenate features only - coordinates remain unchanged
        # This works when all input tensors share the same coordinate structure
        # but have different feature dimensions to combine
        
        # Combine features along the specified dimension
        feats = torch.cat([input.feats for input in inputs], dim=dim)
        
        # Create new sparse tensor using the first input's coordinates
        # but with the concatenated features
        output = inputs[0].replace(feats)

    return output


def sparse_unbind(input: SparseTensor, dim: int) -> List[SparseTensor]:
    """
    Unbind a sparse tensor along a dimension.
    
    Args:
        input (SparseTensor): Sparse tensor to unbind
        dim (int): Dimension to unbind
        
    Returns:
        List[SparseTensor]: List of sparse tensors, each representing a slice along the dimension
    """
    if dim == 0:
        return [input[i] for i in range(input.shape[0])]
    else:
        feats = input.feats.unbind(dim)
        return [input.replace(f) for f in feats]
