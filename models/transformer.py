import torch.nn.functional as F
from torch import Tensor, nn 
import torch 
from collections import OrderedDict
from typing import Optional, Tuple ,List
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import warnings
import math
import numpy as np
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor: 
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Shape:

        - Input: :math:`(N, *, in\_features)` N is the batch size, `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(linear, (input, weight, bias), input, weight, bias=bias)
    return torch._C._nn.linear(input, weight, bias)

def _in_projection(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    b_q: Optional[Tensor] = None,
    b_k: Optional[Tensor] = None,
    b_v: Optional[Tensor] = None,
    ):
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)

    assert w_q.shape == (Eq,Eq)
    assert w_k.shape == (Eq, Ek)
    assert w_v.shape == (Eq,Ev)
    assert b_q is None or b_q.shape == (Eq,)
    assert b_k is None or b_k.shape == (Eq,)
    assert b_v is None or b_v.shape == (Eq,)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)  

def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v: 
        if q is k:
            return linear(q, w, b).chunk(3, dim=-1)  # L,E E,3E   = L,3E 
        else:
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None 
            else:
                b_q, b_kv = b.split([E, E * 2])
            
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)  
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None 
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q,w_q,b_q), linear(k,w_k,b_k), linear(v, w_v, b_v)



class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor) :
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x:torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:bool = True, device = None, dtype = None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)


class MultiheadAttention(nn.Module):

    __constants__ = ['batch_first']  
    bias_k : Optional[torch.Tensor] 
    bias_v: Optional[torch.Tensor]  

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn = False,
                 kdim = None, vdim = None, batch_first = False, device = None, dtype = None):
        #super().__init__()
        factory_kwards = {'device':device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim 
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim 
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim 

        self.num_heads = num_heads 

        self.dropout = dropout 
        self.batch_first = batch_first 
        self.head_dim = embed_dim // num_heads 

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim // num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwards))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwards))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwards))

            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwards))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwards))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias = bias, **factory_kwards)
        if add_bias_kv :
            self.bias_k = Parameter(torch.empty((1,1,embed_dim), **factory_kwards))
            self.bias_v = Parameter(torch.empty((1,1,embed_dim), **factory_kwards))
        else:
            self.bias_k = self.bias_v = None 

        self.add_zero_attn = add_zero_attn 

        self._reset_parameters()
    
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)
    
    def initialize_model_params(self,model):  
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
    
    def Scaled_dot_product_attention(self, q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,):
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.

        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            dropout_p: dropout probability. If greater than 0.0, dropout is applied.

        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.

            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        B, Nt, E = q.shape
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns) 
        attn = torch.bmm(q, k.transpose(-2, -1))  
        if attn_mask is not None:
            attn += attn_mask
        attn = torch.nn.functional.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = torch.nn.functional.dropout(attn, p=dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def Multi_head_attention_forward(self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None
        ):
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
        if has_torch_function(tens_ops):
            return handle_torch_function(
                self.multi_head_attention_forward,
                tens_ops,
                query,
                key,
                value,
                embed_dim_to_check,
                num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k,
                bias_v,
                add_zero_attn,
                dropout_p,
                out_proj_weight,
                out_proj_bias,
                training=training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                static_k=static_k,
                static_v=static_v,
            )
        
        tgt_len, bsz, embed_dim = query.shape

        src_len,_,_ = key.shape 
        assert embed_dim == embed_dim_to_check 

        if isinstance(embed_dim, torch.Tensor):
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        
        assert head_dim * num_heads  == embed_dim 

        if use_separate_proj_weight:
            assert key.shape[:2] == value.shape[:2]
        else:
            assert key.shape == value.shape 

        if not use_separate_proj_weight:
            q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        else:  
            assert q_proj_weight is not None 
            assert k_proj_weight is not None
            assert v_proj_weight is not None 
            if in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = in_proj_bias.chunk(3)
            
            q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

        

        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                attn_mask = attn_mask.to(torch.bool)
        
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool 
            
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")
        

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])  
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1)) 
            if key_padding_mask is not None:
                key_padding_mask = nn.functional.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None
        


        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)   
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)  
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = nn.functional.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = nn.functional.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)


        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        attn_output, attn_output_weights = self.Scaled_dot_product_attention(q, k, v, attn_mask, dropout_p) 
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)   
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None




    def forward (self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: Optional[torch.Tensor]=None,
                 need_weights:bool= True, attn_mask: Optional[Tensor] = None):
        

        if self.batch_first:
            query, key, value = [x.transpose(1,0) for x in (query, key, value)]
        
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = self.Multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, 
                self.in_proj_weight , self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, 
                training = self.training,
                key_padding_mask = key_padding_mask, need_weights = need_weights, 
                attn_mask = attn_mask, use_separate_proj_weight = True, 
                q_proj_weight = self.q_proj_weight, k_proj_weight = self.k_proj_weight,
                v_proj_weight = self.v_proj_weight)
        
        else:
            attn_output, attn_output_weights = self.Multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads, 
                self.in_proj_weight , self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias, 
                training = self.training,
                key_padding_mask = key_padding_mask, need_weights = need_weights, 
                attn_mask = attn_mask)
            
        if self.batch_first:
            return attn_output.transpose(1,0), attn_output_weights 
        else:
            return attn_output, attn_output_weights
        




class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model:int, n_head: int, attn_mask:torch.Tensor = None):
        super().__init__()
        
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        #self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),("gelu",QuickGELU()), ("c_proj", nn.Linear(d_model * 4, d_model ))]))
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)),("gelu",nn.GELU()), ("c_proj", nn.Linear(d_model * 4, d_model ))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype = x.dtype, device = x.device) if self.attn_mask is not None else None 
        #return self.attn(x,x,x, need_weights = False, attn_mask = self.attn_mask)[0]
        return self.attn(x,x,x, need_weights = True, attn_mask = self.attn_mask)  

    def forward(self, x:torch.Tensor):
        tmp, attn = self.attention(self.ln_1(x))
        x = x + tmp
        x = x + self.mlp(self.ln_2(x))
        return x, attn






class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None) :
        super().__init__()
        self.width = width 
        self.layers = layers 
        #self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        #add
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, attn_mask
                )
            for idx in range(layers)
        ])

    #def forward (self, x:torch.Tensor):
    #    return self.resblocks(x)

    def forward(self, x: torch.Tensor, out_layers: list = [3, 6, 9],
                ):
        idx = 0
        out_attn = []
        # out_tokens = x
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            if idx == 12:
                x, attn = r(x)
                out_attn.append(attn)
            else:
                x, attn_tmp = r(x)
            if idx in out_layers:
                out_tokens.append(x)
                # out_tokens = x
        return x, out_attn, out_tokens


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int , patch_size: int , width: int, layers: int , heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution 
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= width, kernel_size = patch_size, stride= patch_size, bias=False)

        scale = width ** -0.5  #   
        self.class_embedding =  nn.Parameter(scale * torch.randn(width))

        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) **2 + 1, width))
        self.ln_pre = LayerNorm(width)   

        self.transformer = Transformer( width, layers, heads) 

        self.ln_post = LayerNorm(width)
        
        #add
        self.patch_size = patch_size
        self.grid_size = input_resolution // patch_size

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x:torch.Tensor, out_layers: list):
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0,2,1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],dim = 1) 

        x = x + self.positional_embedding.to(x.dtype)
        #add 
        #x = x + self.positional_embedding.to(x.dtype)[:(140 // self.patch_size) **2 + 1,:]

        x = self.ln_pre(x)

        x = x.permute(1,0,2)

        #x = self.transformer(x)
        x, attn, patch_tokens = self.transformer(x, out_layers)
        # attn = attn[0, 0, 1:].view(14, 14)  # 49
        B, C, L = attn[0].shape
        H = int(np.sqrt(L-1))
        out_attn = torch.zeros([H, H]).to('cuda')
        for i in range(len(attn)):
            out_attn += attn[i][0, 0, 1:].view(H, H)

        x = x.permute(1,0,2)
        #add
        patch_tokens = [patch_tokens[t].permute(1, 0, 2) for t in range(len(patch_tokens))]  # LND -> NLD

        x = self.ln_post(x[:, 0, :])   

        if self.proj is not None:
            x = x @ self.proj   
        return x, patch_tokens



  


if __name__ == '__main__':
    '''
    input = torch.rand((1,3,224,224),dtype = torch.float32)
    model = VisionTransformer(224, 16, 728, 2, 2, 1024)
    print(input.size())
    result = model(input)
    print(result.size())
    print(result)
    '''

    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC
    def _transform(n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    transform = _transform((336,336))
    #image2 = transform(Image.open("Dog.jpeg")).unsqueeze(0).to(device)
    #image3 = transform(Image.open("Cat.jpg")).unsqueeze(0).to(device)

    image2 = torch.rand((1,3,336,336),dtype = torch.float32).to(device)
    image3 = torch.rand((1,3,336,336),dtype = torch.float32).to(device)

    model = VisionTransformer(336, 16, 728, 12, 2, 1024).to(device)
    with torch.no_grad():
        print(torch.equal(image2,image3))   

        image_features3,_ = model(image3, [3,6,9])

        image_features2,_ = model(image2,[3,6,9])
        print(torch.equal(image_features2,image_features3))









        



