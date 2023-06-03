import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial pyramid pooling implementation

class SPP(nn.Module):

    def __init__(self, levels = 3, pool_type = 'max_pool'):
        super(SPP, self).__init__()

        if pool_type not in ['max_pool', 'avg_pool', 'max_avg_pool']:
            raise ValueError('Unknown pool_type. Must be either \'max_pool\', \'avg_pool\' or both')
        
        self.pooling_output_size = sum([4 ** level for level in range(levels)]) * 512

        self.levels = levels
        self.pool_type = pool_type

    def forward(self, input_x):
        out = self._spatial_pyramid_pooling(input_x, self.levels)
        return out
    
    def _pyramid_pooling(self, input_x, output_sizes):
        pyramid_level_tensors = []
        for tsize in output_sizes:
            if self.pool_type == 'max_pool':
                pyramid_level_tensor = F.adaptive_max_pool2d(input_x, tsize)
                pyramid_level_tensor = pyramid_level_tensor.view(input_x.size(0), -1)
            if self.pool_type == 'avg_pool':
                pyramid_level_tensor = F.adaptive_avg_pool2d(input_x, tsize)
                pyramid_level_tensor = pyramid_level_tensor.view(input_x.size(0), -1)
            if self.pool_type == 'max_avg_pool':
                pyramid_level_tensor_max = F.adaptive_max_pool2d(input_x, tsize)
                pyramid_level_tensor_max = pyramid_level_tensor_max.view(input_x.size(0), -1)
                pyramid_level_tensor_avg = F.adaptive_avg_pool2d(input_x, tsize)
                pyramid_level_tensor_avg = pyramid_level_tensor_avg.view(input_x.size(0), -1)
                pyramid_level_tensor = torch.cat([pyramid_level_tensor_max, pyramid_level_tensor_avg], dim=1)

            pyramid_level_tensors.append(pyramid_level_tensor)

        return torch.cat(pyramid_level_tensors, dim=1)

    def _spatial_pyramid_pooling(self, input_x, levels):
        output_sizes = [(int( 2 **level), int( 2 **level)) for level in range(levels)]
        return self._pyramid_pooling(input_x, output_sizes)