import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

# Helper functions
def to_2tuple(x):
    return (x, x) if isinstance(x, int) else x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.2):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        # 随机选择哪些路径丢弃
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        random_tensor = torch.floor(random_tensor)  # 将随机数阈值化
        output = x / keep_prob * random_tensor  # 将被丢弃的路径设置为零

        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (B*T, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=proj_drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, T, H*W, C)
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, T, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, T, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            attn_mask = mask_matrix.type(x.dtype)
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # merge windows
        x = x.view(B, T, Hp * Wp, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Helper Functions: window partition and reverse (for handling windows)
def window_partition(x, window_size):
    """
    Partition input tensor into windows of shape [B, window_size, window_size, C]
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size, window_size, W // window_size, window_size).contiguous()
    windows = x.view(B, T, H // window_size, W // window_size, window_size, window_size, C)
    windows = windows.permute(0, 1, 2, 3, 5, 4, 6)  # Swap last two axes to shape (B, T, grid_h, grid_w, window_size, window_size, C)
    return windows.view(-1, window_size * window_size, C)  # Flatten the window to shape (B*T*grid_h*grid_w, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    """
    Reverse the window partition operation.
    """
    B_T, window_area, C = windows.shape
    grid_h = H // window_size
    grid_w = W // window_size
    windows = windows.view(B_T, grid_h, grid_w, window_size, window_size, C)

    windows = windows.permute(0, 1, 2, 4, 3, 5)  # Swap back the axes to (B, T, grid_h, grid_w, window_size, window_size, C)
    return windows.contiguous().view(-1, H, W, C)  # Reshape back to the original shape (B, T, H, W, C)


# Swin Transformer for Sequence Tasks (including time dimension)
class SwinTransformer3D(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, depths, window_size=7, mlp_ratio=4., num_classes=155):
        super(SwinTransformer3D, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Initial Convolution
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Define the Swin Transformer Blocks
        self.layers = nn.ModuleList()
        for depth in depths:
            self.layers.append(SwinTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,  # Set shift size to half of window size for SW-MSA
                mlp_ratio=mlp_ratio
            ))

        # MLP head for classification
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Initial embedding
        x = self.conv1(x)  # (B, T, embed_dim, H, W)
        x = x.view(B, T, H * W, self.embed_dim)  # Reshape to (B, T, H*W, embed_dim)

        # Pass through Swin Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask_matrix=None)  # mask_matrix could be added for attention masking if needed

        # Global average pooling across the spatial dimensions (H * W) for classification
        x = x.mean(dim=2)  # Averaging over H*W
        x = self.fc(x)  # (B, T, num_classes)

        return x

# Example Usage
# Assuming your input shape is (B, T, C, H, W) where B is the batch size, T is the time frames, C is the channels,
# H and W are the spatial dimensions (height and width) of the input.

input_tensor = torch.randn(8, 300, 3, 17, 17)  # Example shape: (B=8, T=300, C=3, H=17, W=17)
model = SwinTransformer3D(in_channels=3, embed_dim=128, num_heads=4, depths=[2, 2, 6, 2], num_classes=155)

output = model(input_tensor)  # Output shape will be (B, T, num_classes)
print(output.shape)  # Expected: (8, 300, 155)
