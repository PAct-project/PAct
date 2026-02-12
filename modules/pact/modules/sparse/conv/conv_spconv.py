import torch
import torch.nn as nn
from .. import SparseTensor
from .. import DEBUG
from . import SPCONV_ALGO

class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
        super(SparseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        algo = None
        if SPCONV_ALGO == 'native':
            algo = spconv.ConvAlgo.Native
        elif SPCONV_ALGO == 'implicit_gemm':
            algo = spconv.ConvAlgo.MaskImplicitGemm
        if stride == 1 and (padding is None):
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
        else:
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
        self.padding = padding

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
        new_data = self.conv(x.data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout

        if spatial_changed and (x.shape[0] != 1):
            # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
            fwd = new_data.indices[:, 0].argsort()
            bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
            sorted_feats = new_data.features[fwd]
            sorted_coords = new_data.indices[fwd]
            unsorted_data = new_data
            new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )

        if spatial_changed and (x.shape[0] != 1):
            out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
            out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)
 
        return out


class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

    def forward(self, x: SparseTensor) -> SparseTensor:
        spatial_changed = any(s != 1 for s in self.stride)
        if spatial_changed:
            # recover the original spconv order
            data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
            bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
            data = data.replace_feature(x.feats[bwd])
            if DEBUG:
                assert torch.equal(data.indices, x.coords[bwd]), 'Recover the original order failed'
        else:
            data = x.data

        new_data = self.conv(data)
        new_shape = [x.shape[0], self.conv.out_channels]
        new_layout = None if spatial_changed else x.layout
        out = SparseTensor(
            new_data, shape=torch.Size(new_shape), layout=new_layout,
            scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
            spatial_cache=x._spatial_cache,
        )
        return out


# Global pooling implementations that wrap spconv's implementations to preserve autograd
class SparseGlobalMaxPool3d(nn.Module):
    """Global max pooling for SparseTensor using spconv.pytorch.pool.SparseGlobalMaxPool.
    Returns per-batch (B, C) features and wraps them into a SparseConvTensor with
    one voxel per batch so it fits the project's SparseTensor wrapper.
    """
    def __init__(self):
        super(SparseGlobalMaxPool3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        from spconv.pytorch.pool import SparseGlobalMaxPool as _SPGlobalMax
        self._spconv = spconv
        self.pool = _SPGlobalMax()

    def forward(self, x: SparseTensor) -> SparseTensor:
        # spconv pool expects a SparseConvTensor; x.data is that object for spconv backend
        data = x.data
        ops = self._spconv.ops
        out_indices, counts = ops.global_pool_rearrange(data.indices, data.batch_size)
        counts_np = counts.cpu().numpy()
        # Use ops.global_pool_rearrange to compute per-batch per-channel max robustly
        B = int(x.shape[0])
        C = int(x.shape[1])
        res_list = []
        for i in range(data.batch_size):
            real_inds = out_indices[i, :counts_np[i]]
            if real_inds.numel() == 0:
                res_list.append(torch.zeros(C, device=data.features.device, dtype=data.features.dtype))
            else:
                real_features = data.features[real_inds]
                res_list.append(real_features.max(dim=0).values)
        out_feats = torch.stack(res_list, dim=0)

        B = int(x.shape[0])
        coord_tensor = getattr(x, 'coords', x.data.indices)
        coord_dtype = coord_tensor.dtype
        coord_device = coord_tensor.device
        coords_dim = coord_tensor.shape[1]  # 1 + spatial_dim

        batch_idx = torch.arange(B, device=coord_device, dtype=coord_dtype).unsqueeze(1)
        zeros = coord_tensor.new_zeros((B, coords_dim - 1))
        out_coords = torch.cat([batch_idx, zeros], dim=1)

        spatial_shape = tuple([1] * (coords_dim - 1))
        new_data = self._spconv.SparseConvTensor(out_feats, out_coords, spatial_shape, B)  # type: ignore

        C = int(out_feats.shape[1])
        out = SparseTensor(
            new_data, shape=torch.Size([B, C]), layout=None,
            scale=x._scale,
            spatial_cache=x._spatial_cache,
        )
        return out


class SparseGlobalAvgPool3d(nn.Module):
    """Global avg pooling for SparseTensor using spconv.pytorch.pool.SparseGlobalAvgPool."""
    def __init__(self):
        super(SparseGlobalAvgPool3d, self).__init__()
        if 'spconv' not in globals():
            import spconv.pytorch as spconv
        from spconv.pytorch.pool import SparseGlobalAvgPool as _SPGlobalAvg
        self._spconv = spconv
        self.pool = _SPGlobalAvg()

    def forward(self, x: SparseTensor) -> SparseTensor:
        data = x.data
        ops = self._spconv.ops
        out_indices, counts = ops.global_pool_rearrange(data.indices, data.batch_size)
        counts_np = counts.cpu().numpy()
        # Compute per-batch per-channel mean using ops.global_pool_rearrange
        B = int(x.shape[0])
        C = int(x.shape[1])
        res_list = []
        for i in range(data.batch_size):
            real_inds = out_indices[i, :counts_np[i]]
            if real_inds.numel() == 0:
                res_list.append(torch.zeros(C, device=data.features.device, dtype=data.features.dtype))
            else:
                real_features = data.features[real_inds]
                res_list.append(real_features.mean(dim=0))
        out_feats = torch.stack(res_list, dim=0)

        coord_tensor = getattr(x, 'coords', x.data.indices)
        coord_dtype = coord_tensor.dtype
        coord_device = coord_tensor.device
        coords_dim = coord_tensor.shape[1]

        batch_idx = torch.arange(B, device=coord_device, dtype=coord_dtype).unsqueeze(1)
        zeros = coord_tensor.new_zeros((B, coords_dim - 1))
        out_coords = torch.cat([batch_idx, zeros], dim=1)

        spatial_shape = tuple([1] * (coords_dim - 1))
        new_data = self._spconv.SparseConvTensor(out_feats, out_coords, spatial_shape, B)  # type: ignore

        C = int(out_feats.shape[1])
        out = SparseTensor(
            new_data, shape=torch.Size([B, C]), layout=None,
            scale=x._scale,
            spatial_cache=x._spatial_cache,
        )
        return out
