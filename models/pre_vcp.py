import torch 
import torch.nn as nn
import numpy as np 

'''
class Linear1(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, k = 5):
        super(Linear1, self).__init__()
        self.fc = nn.ModuleList([nn.Sequential(nn.Linear(dim_in, dim_out)) for i in range(k)])
        self.num_layer = k
    def forward(self, tokens):
        
        token_list = []
        for i in range(self.num_layer):
            tokens = self.fc[i](tokens)
            token_list.append(tokens)
        return torch.cat(token_list, dim = 1)
'''
class Linear1(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, k = 5):
        super(Linear1, self).__init__()
        self.fc = nn.Conv1d(1, k, 3, stride=1, padding="same")
        self.num_layer = k
    def forward(self, tokens):
        
        result = self.fc(tokens)
        return result

class Context_Prompting(nn.Module):
    def __init__(self, model_config, cla_len):
        super().__init__()
        assert model_config['text_cfg']['width'] == model_config['embed_dim']
        self.prompt_query = nn.Parameter(torch.randn(1, cla_len, model_config['text_cfg']['width']))
        self.cla_len = cla_len
        self.prompt_linear1 = Linear1(model_config['text_cfg']['width'], model_config['text_cfg']['width'] + 256, model_config['text_cfg']['width'], k = cla_len)
        self.prompt_temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        nn.init.trunc_normal_(self.prompt_query)
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
    


    def before_extract_feat(self, x, img_feature, use_global = True):
        B, C = img_feature.shape
        global_feat = img_feature
        global_feat_new = self.prompt_linear1(global_feat.reshape(B, 1, C))
        prompt_query = self.prompt_query + torch.zeros((B, self.prompt_query.shape[-2], self.prompt_query.shape[-1]), dtype=self.prompt_query.dtype, device=self.prompt_query.device)
        if use_global:
            class_feature =  prompt_query  +  global_feat_new 
        else:
            class_feature = prompt_query 
        return class_feature 
    



