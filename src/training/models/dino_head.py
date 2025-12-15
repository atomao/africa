import torch
import torch.nn as nn
import torch.nn.functional as F


class BNUHeadBase(nn.Module):
    """Base Batch Norm U-Net style decode head with upsampling."""
    
    def __init__(
        self,
        in_channels: list[int],
        in_index: list[int],
        intermediate_channels: list[int],
        num_classes: int = 3,
        align_corners: bool = False,
        dropout_ratio: float = 0.0,
        resize_factors: list[float] | None = None,
        input_transform: str = "resize_concat",
        use_sync_bn: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_index = in_index
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.resize_factors = resize_factors
        self.input_transform = input_transform
        
        # Calculate total input channels after concatenation
        self.total_in_channels = sum(in_channels[i] for i in in_index)
        
        # Initial batch norm - use SyncBatchNorm to match old code
        if use_sync_bn:
            self.bn = nn.SyncBatchNorm(self.total_in_channels)
        else:
            self.bn = nn.BatchNorm2d(self.total_in_channels)
        
        # Build progressive refinement layers with UPSAMPLING
        self.conv_blocks = nn.ModuleList()
        self.upsample_scales = []  # Store upsampling factors
        
        in_ch = self.total_in_channels
        for i, out_ch in enumerate(intermediate_channels):
            if i == 0:
                # First layer is 1x1 conv
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            else:
                # Rest are 3x3 convs with padding
                conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                # Add upsampling for all blocks except the first
                self.upsample_scales.append(2.0)
            
            self.conv_blocks.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # Final upsampling before classification
        self.upsample_scales.append(2.0)
        
        # Final classification layer
        self.conv_seg = nn.Conv2d(in_ch, num_classes, kernel_size=3, padding=1)
        
        # Dropout (not in original, but can be kept as optional)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        
    def _transform_inputs(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Transform inputs matching the old implementation logic."""
        
        if self.input_transform == "resize_concat":
            # Handle list inputs (for cls token) - flatten nested lists
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            
            # Handle 2D tensors by expanding to 4D
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            
            # Select indices
            inputs = [inputs[i] for i in self.in_index]
            
            # Apply resize factors if specified
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), \
                    f"Mismatch: {len(self.resize_factors)} != {len(inputs)}"
                inputs = [
                    F.interpolate(
                        x, 
                        scale_factor=f, 
                        mode="bilinear" if f >= 1 else "area",
                        align_corners=self.align_corners if f >= 1 else None
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
            
            # Resize all to the same size (size of first input)
            target_size = inputs[0].shape[2:]
            upsampled_inputs = [
                F.interpolate(
                    x, 
                    size=target_size, 
                    mode="bilinear", 
                    align_corners=self.align_corners
                )
                if x.shape[2:] != target_size else x
                for x in inputs
            ]
            
            # Concatenate along channel dimension
            return torch.cat(upsampled_inputs, dim=1)
            
        elif self.input_transform == "multiple_select":
            return [inputs[i] for i in self.in_index]
        else:
            return inputs[self.in_index[0]]
    
    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass with progressive upsampling."""
        # Transform and concatenate inputs
        x = self._transform_inputs(inputs)
        
        # Apply initial batch norm
        x = self.bn(x)
        
        # Progressive refinement through conv blocks with upsampling
        for i, conv_block in enumerate(self.conv_blocks):
            # Apply convolution block
            x = conv_block(x)
            
            # Apply upsampling after each block except the first
            if i > 0 and i - 1 < len(self.upsample_scales):
                x = F.interpolate(
                    x, 
                    scale_factor=self.upsample_scales[i-1], 
                    mode='nearest'
                )
        
        # Apply final upsampling before classification
        if len(self.conv_blocks) > 0 and len(self.upsample_scales) > len(self.conv_blocks) - 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_scales[-1],
                mode='nearest'
            )
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification
        x = self.conv_seg(x)
        
        return x


class BN_U_Head_768(BNUHeadBase):
    """Standard BN U-Net head with default channel progression."""
    
    def __init__(
        self, 
        in_channels: list[int], 
        in_index: list[int], 
        num_classes: int = 3,
        resize_factors: list[float] | None = None,
        **kwargs
    ):
        # Match the old layer_channels: [768, 256, 128, 64, 32]
        intermediate_channels = [768, 256, 128, 64, 32]
        
        super().__init__(
            in_channels=in_channels,
            in_index=in_index,
            intermediate_channels=intermediate_channels,
            num_classes=num_classes,
            resize_factors=resize_factors,
            **kwargs
        )