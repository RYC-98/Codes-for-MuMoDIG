import torch
import torch.nn as nn

class LBQuantization(nn.Module):
    def __init__(self, region_num, transforms_like=False):
        """
        region_num: int;
        """
        super().__init__()
        self.region_num = region_num
        self.transforms_like = transforms_like

    def get_params(self, x):
        """
        x: (C, H, W)·
        returns (C), (C), (C)
        """
        C, _, _ = x.size() # one batch img
        min_val, max_val = x.reshape(C, -1).min(1)[0], x.reshape(C, -1).max(1)[0] 


        total_region_percentile_number = (torch.ones(C) * (self.region_num - 1)).int() 

        return min_val, max_val, total_region_percentile_number

    def forward(self, x):
        """
        x: (B, c, H, W) or (C, H, W)
        """
        EPSILON = 1

        if not self.transforms_like:
            B, c, H, W = x.shape
            C = B * c                  
            x = x.reshape(C, H, W)
        else:
            C, H, W = x.shape
        min_val, max_val, total_region_percentile_number_per_channel = self.get_params(x) 


        # region percentiles for each channel
        region_percentiles = torch.rand(total_region_percentile_number_per_channel.sum(), device=x.device)

        region_percentiles_per_channel = region_percentiles.reshape([-1, self.region_num - 1])

        # ordered region ends 
        region_percentiles_pos = (region_percentiles_per_channel * (max_val - min_val).reshape(C, 1) + min_val.reshape(C, 1)).reshape(C, -1, 1, 1)
        
        ordered_region_right_ends_for_checking = torch.cat([region_percentiles_pos, max_val.reshape(C, 1, 1, 1)+EPSILON], dim=1).sort(1)[0]
        ordered_region_right_ends = torch.cat([region_percentiles_pos, max_val.reshape(C, 1, 1, 1)+1e-6], dim=1).sort(1)[0]
        
        ordered_region_left_ends = torch.cat([min_val.reshape(C, 1, 1, 1), region_percentiles_pos], dim=1).sort(1)[0]

        is_inside_each_region = (x.reshape(C, 1, H, W) < ordered_region_right_ends_for_checking) * (x.reshape(C, 1, H, W) >= ordered_region_left_ends) # -> (C, self.region_num, H, W); boolean
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range

        associated_region_id = torch.argmax(is_inside_each_region.int(), dim=1, keepdim=True)  # -> (C, 1, H, W)


        proxy_vals = torch.gather(ordered_region_left_ends.expand([-1, -1, H, W]), 1, associated_region_id)[:,0]
        x = proxy_vals.type(x.dtype)



        if not self.transforms_like:
            x = x.reshape(B, c, H, W)   

        return x

