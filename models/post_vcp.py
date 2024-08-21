
import torch 
from torch import Tensor, nn
from typing import Tuple, Type
import numpy as np



class Zero_Parameter(nn.Module):

    def __init__(self, dim_v,dim_t, dim_out, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop = 0., proj_drop = 0.):
        super().__init__()
        self.num_heads = num_heads 
        self.head_dim = dim_out // num_heads
        self.dim_out = dim_out

        self.scale = qk_scale or dim_out ** -0.5
        
        self.q_proj_pre = nn.Conv1d(dim_t, dim_out, kernel_size=1)
        self.k_proj_pre_1 =nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.v_proj_pre_1 = nn.Conv1d(dim_v, dim_out, kernel_size=1)
        self.proj_post_t = nn.Conv1d(dim_out, dim_out, kernel_size=1)
        
        self.prompt_temp_l1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.beta_t = 1 

    
        self._initialize_weights()


    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(self, F_t, F_s):
        B1, N1, C1 = F_t.shape
        B2, N2, C2 = F_s.shape
        assert B1 == B2
        q_t = self.q_proj_pre(F_t.permute(0, 2, 1)).permute(0, 2, 1).reshape(B1, N1, self.num_heads, self.head_dim)  #1
        k_s = self.k_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  #1
        v_s = self.v_proj_pre_1(F_s.permute(0, 2, 1)).permute(0, 2, 1).reshape(B2, N2, self.num_heads, self.head_dim)  #1
        attn_t = torch.einsum('bnkc,bmkc->bknm', q_t, k_s) * self.beta_t
        attn_t = attn_t.softmax(dim = -1)
        F_t_a = torch.einsum('bknm,bmkc->bnkc', attn_t, v_s).reshape(B1, N1, self.dim_out)
        F_t_a = self.proj_post_t(F_t_a.permute(0, 2, 1)).permute(0, 2, 1) 
        F_t_a = F_t_a / F_t_a.norm(dim=-1, keepdim = True)

        return F_t_a, F_t_a


