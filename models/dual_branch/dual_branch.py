import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from datasets.processor import SpecAug
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

class SqueezeAndExcitationBlock(nn.Module):
    def __init__(self, num_channels, ratio=16, dim_num=2, dim='c') -> None:
        super().__init__()
        self.num_channels = num_channels        
        self.dim_num = dim_num      
        if len(dim) >= 4:
            dim = dim[3:]            
        else:
            dim = 'c'
        self.dim = dim
        if dim == 'c':
            ratio = 16
        self.ratio = ratio    
        #print('ratio: ',ratio)
        #print('dim:   ',dim)
        
        if dim_num == 1:    # (batch_size, c, f)
            if dim == 'c':
                self.permute_list = (0, 1, 2)   # (batch_size, c, f)
                self.ipermute_list = (0, 1, 2)
            elif dim == 'f':
                self.permute_list = (0, 2, 1)   # (batch_size, f, c)
                self.ipermute_list = (0, 2, 1)
        elif dim_num == 2:  # (batch_size, c, t, f)
            if dim == 't':
                self.permute_list = (0, 2, 1, 3)   # (batch_size, t, c, f)
                self.ipermute_list = (0, 2, 1, 3)
            elif dim == 'f':
                self.permute_list = (0, 3, 1, 2)   # (batch_size, f, c, t)
                self.ipermute_list = (0, 2, 3, 1)
            elif dim == 'c':
                self.permute_list = (0, 1, 2, 3)   # (batch_size, c, t, f)
                self.ipermute_list = (0, 1, 2, 3)
        
        if self.dim_num==2:
            self.L1 = nn.AdaptiveAvgPool2d(1)   # tf.keras.layers.GlobalAveragePooling2D()
            # input (B,C,H,W) output (B,C,1,1)
        elif self.dim_num == 1:
            self.L1 = nn.AdaptiveAvgPool1d(1)   # tf.keras.layers.GlobalAveragePooling1D()
            # input (B,C,L) output (B,C,1)
        
        self.L2 = nn.Sequential(
            nn.Linear(num_channels, num_channels // ratio, bias=False),
            nn.ReLU()            
        )
        # self.L2 = tf.keras.layers.Dense(self.num_channels//self.ratio, activation='relu', use_bias=False)
        
        self.L3 = nn.Sequential(
            nn.Linear(num_channels // ratio, num_channels, bias=False),
            nn.Sigmoid()
        )
        # self.L3 = tf.keras.layers.Dense(self.num_channels, activation='sigmoid', use_bias=False)
        # self.L4 = tf.keras.layers.Multiply()
    
    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:              
        #input = input.permute(*self.permute_list)
        
        x = self.L1(input).squeeze(-1)        
        if self.dim_num==2:
            x = x.squeeze(-1)
        
        x = self.L2(x)
        x = self.L3(x).unsqueeze(2)
        if self.dim_num==2:
            x = x.unsqueeze(3)
            
        output = torch.mul(input,x) # input (B,C,L) / (B,C,H,W) x (B,C,1,1)
        # mul is to calculate elements under each same index
        # due to broadcasting, output has the same size with input
        
        #output = output.permute(*self.ipermute_list)
        return output


# Attention module is not in the origin work of Kevin Wilkinghoff
def get_nstate(i: int, feat: str = 'f', trum0_or_gram1: bool = 1) -> int:
    if not trum0_or_gram1:
        list = [128,128,128] if (feat=='f' or feat=='se') else [1250,40,10] if feat=='c' else [128,128,128]
    else:
        list = [77,39,20,10] if feat=='f' else [128,64,32,16] if (feat=='c' or feat=='c_p' or feat=='t') else [16,32,64,128] if feat=='se' else [77,39,20,10]
    out = list[i]
    return out

class Channel_Attention(nn.Module): # special for dim= 'c_p', dim_num = 4
    # Note that for (batch_size, n_ctx, n_state), attention mechanism is applied on n_ctx dim rather than n_state dim !       
    # Note that Channel_Attention has no linear head for V and out, to resemble the function of SE
    def __init__(self, n_state: int, n_head: int, with_head: bool = True):
        super().__init__()
        self.n_head = n_head
        
        self.query = nn.Linear(n_state, n_state) if with_head else nn.Identity()
        self.key = nn.Linear(n_state, n_state, bias=False) if with_head else nn.Identity()
        #self.value = nn.Linear(n_state, n_state) if with_head else nn.Identity()
        #self.out = nn.Linear(n_state, n_state)
        
        # (batch_size, c, t, f)
        self.permute_list = (0, 1, 3, 2)   
        self.ipermute_list = (0, 1, 3, 2)
         
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        return_permuted: bool = False
    ):        
        v = x if xa is None else xa     # (batch_size, c, t, f)        
        v = v.permute(*self.permute_list).flatten(start_dim=2)      # (batch_size, c, f*t)
        # Keep in mind that t is inferior to f, which is important in 'deflatten' process
        
        x = x.permute(*self.permute_list)       # (batch_size, c, f, t)
        x = torch.mean(x, dim=3, keepdim=False) # (batch_size, c, f)
        if xa is not None:
            xa = xa.permute(*self.permute_list)
            xa = torch.mean(xa, dim=3, keepdim=False)                
        q = self.query(x)
        k = self.key(x if xa is None else xa)

        wv = self.qkv_attention(q, k, v, mask)  # (batch_size, c, f, t)        
        if return_permuted:
            return wv
        else:
            return wv.permute(*self.ipermute_list)      # (batch_size, c, t, f)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_c, n_f = q.shape
        scale = (n_f // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale       # (batch, head, n_c, n_f/head)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)               # (batch, head, n_c, n_f*n_t/head)

        qk = q @ k      # (batch, head, n_c, n_c)
        if mask is not None:
            qk = qk + mask[:n_c, :n_c].unsqueeze(0).unsqueeze(0)
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)        
        # w @ v (batch, head, n_c, n_f*n_t/head)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)     # (batch, n_c, n_f*n_t)        
        wv = wv.view(n_batch, n_c, n_f, -1)
        return wv

class Attention(nn.Module):
    # Note that for (batch_size, n_ctx, n_state), attention mechanism is applied on n_ctx dim rather than n_state dim !    
    def __init__(self, n_state: int, n_head: int, dim: str = 'f', dim_num: int = 4, with_head: bool = True):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.dim_num = dim_num
        self.use_Channel_Attention = (dim=='c_p')
        self.use_SE = (len(dim)>=2) and (dim[0:2]=='se')
        
        if not((dim == 't' and dim_num == 4) or dim == 'f' or dim == 'c' or (dim == 'c_p' and dim_num == 4) or self.use_SE):
            raise ValueError(f"Attention: requested dim is {dim} in {dim_num} dims, not supported")
        if not(dim_num == 4 or dim_num == 3):
            raise ValueError(f"Attention: input dim_num is {dim_num}. Only 4 / 3 are supported")
        
        if self.use_Channel_Attention:
            self.ThrowPot = Channel_Attention(n_state,n_head,with_head)
        elif self.use_SE:        
            self.ThrowPot = SqueezeAndExcitationBlock(n_state, n_head, dim_num=(dim_num-2), dim=dim)
        else:
            self.query = nn.Linear(n_state, n_state) if with_head else nn.Identity()
            self.key = nn.Linear(n_state, n_state, bias=False) if with_head else nn.Identity()
            self.value = nn.Linear(n_state, n_state) if with_head else nn.Identity()
            self.out = nn.Linear(n_state, n_state) if with_head else nn.Identity()   
            
            if dim_num == 3:    # (batch_size, c, f)
                #self.out = nn.Linear(n_state, n_state)
                if dim == 'c':
                    self.permute_list = (0, 1, 2)   # (batch_size, c, f)
                    self.ipermute_list = (0, 1, 2)
                elif dim == 'f':
                    self.permute_list = (0, 2, 1)   # (batch_size, f, c)
                    self.ipermute_list = (0, 2, 1)
            elif dim_num == 4:  # (batch_size, c, t, f)
                if dim == 't':
                    self.permute_list = (0, 1, 2, 3)   # (batch_size, c, t, f)
                    self.ipermute_list = (0, 1, 2, 3)
                elif dim == 'f':
                    self.permute_list = (0, 1, 3, 2)   # (batch_size, c, f, t)
                    self.ipermute_list = (0, 1, 3, 2)
                elif dim == 'c':
                    self.permute_list = (0, 2, 1, 3)   # (batch_size, t, c, f)
                    self.ipermute_list = (0, 2, 1, 3)
          
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        return_permuted: bool = False
    ):        
        if self.use_Channel_Attention:
            return self.ThrowPot(x,xa,mask,return_permuted)
        elif self.use_SE:
            return self.ThrowPot(x)       
        else:
            x = x.permute(*self.permute_list)        
            if xa is not None:
                xa = xa.permute(*self.permute_list)
            
            q = self.query(x)
            if kv_cache is None or xa is None or self.key not in kv_cache:
                # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
                # otherwise, perform key/value projections for self- or cross-attention as usual.                
                k = self.key(x if xa is None else xa)
                v = self.value(x if xa is None else xa)
            else:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = kv_cache[self.key]
                v = kv_cache[self.value]

            wv = self.qkv_attention(q, k, v, mask)
            wv = self.out(wv)           
            if return_permuted:
                return wv
            else:
                return wv.permute(*self.ipermute_list)

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        if self.dim_num == 4:
            n_batch, n_ctx, n_index, n_state = q.shape
            scale = (n_state // self.n_head) ** -0.25
            q = q.view(*q.shape[:3], self.n_head, -1).permute(0, 3, 1, 2, 4) * scale    # (batch, head, ctx, index, state/head)
            k = k.view(*k.shape[:3], self.n_head, -1).permute(0, 3, 1, 4, 2) * scale
            v = v.view(*v.shape[:3], self.n_head, -1).permute(0, 3, 1, 2, 4)

            qk = q @ k      # (batch, head, ctx, index, index)
            if mask is not None:
                qk = qk + mask[:n_index, :n_index].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 3, 1, 4).flatten(start_dim=3)
        
        elif self.dim_num == 3:
            n_batch, n_ctx, n_state = q.shape
            scale = (n_state // self.n_head) ** -0.25
            q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale       # (batch, head, ctx, state/head)
            k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
            v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

            qk = q @ k      # (batch, head, ctx, ctx)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx].unsqueeze(0).unsqueeze(0)
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

class Res_Attention(nn.Module):
    # attn with res+ (optional) ffn with res     
    def __init__(self, n_state: int, n_head: int, dim: str = 'f', dim_num: int = 4, with_head: bool = True,
                 BN0_or_LN1: bool = 0, input_shape: int = [1,128,10,16],
                 use_FFN: bool = False, exp_ratio: int = 2, activation: str = 'relu'):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.dim_num = dim_num
        self.use_FFN = use_FFN
        #self.activation = activation
        
        self.attn = Attention(n_state, n_head, dim, dim_num, with_head)
        if not self.dim == 'se':
            if BN0_or_LN1:
                norm_dim = [input_shape[self.attn.permute_list[-2]], input_shape[self.attn.permute_list[-1]]]
            else:
                norm_dim = input_shape[self.attn.permute_list[1]]
            if dim_num == 4:
                self.norm1 = nn.LayerNorm(norm_dim) if BN0_or_LN1 else nn.BatchNorm2d(norm_dim)
            else:
                self.norm1 = nn.LayerNorm(norm_dim) if BN0_or_LN1 else nn.BatchNorm1d(norm_dim)
            if use_FFN:
                self.FFN = nn.Sequential(
                    nn.Linear(n_state, n_state*exp_ratio),
                    nn.GELU() if (activation == 'gelu' or activation == 'GELU')
                    else nn.ReLU(),
                    nn.Linear(n_state*exp_ratio, n_state)
                )
                if dim_num == 4:
                    self.norm2 = nn.LayerNorm(norm_dim) if BN0_or_LN1 else nn.BatchNorm2d(norm_dim)
                else:
                    self.norm2 = nn.LayerNorm(norm_dim) if BN0_or_LN1 else nn.BatchNorm1d(norm_dim)
    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):         
        if self.dim == 'se':
            return self.attn(x,xa,mask,kv_cache)
        x_m = self.attn(x,xa,mask,kv_cache, return_permuted=True)
        x_r = x.permute(*self.attn.permute_list)
        x_r = self.norm1(x_m + x_r)
        if self.use_FFN:
            x_m = self.FFN(x_r)
            x_r = self.norm2(x_m + x_r)
        return x_r.permute(*self.attn.ipermute_list)

class net_trum(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        attn_dim: list[str] = ['se','se','se'],
        attn_nhead: list[int] = [4,4,4],
        attn_on_dpth: bool = False
    ) -> None:       
        super().__init__()
        self.use_bias = use_bias
        self.attn_dim = attn_dim
        self.attn_nhead = attn_nhead       
        self.attn_on_dpth = attn_on_dpth
        
        if attn_on_dpth:
            self.flatten1 = nn.AdaptiveAvgPool1d(1)
            self.flatten2 = nn.AdaptiveAvgPool1d(1)
            self.flatten3 = nn.AdaptiveAvgPool1d(1)
            self.LN_AoD = nn.LayerNorm(128)
            self.AoD = Attention(n_state=128, n_head=4, dim='c', dim_num=3)
            self.pool_AoD = nn.Sequential(nn.BatchNorm1d(128), nn.ReLU())
            self.out_AoD = nn.Sequential(
                                nn.Linear(256, 128, bias=use_bias),
                                nn.BatchNorm1d(128),
                                nn.ReLU())
                        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=256, stride=64, padding=96, bias=use_bias),           
            nn.ReLU(),
            Attention(get_nstate(0,attn_dim[0],0),attn_nhead[0],dim=attn_dim[0],dim_num=3)
            #Res_Attention(get_nstate(0,attn_dim[0],0),attn_nhead[0],dim=attn_dim[0],dim_num=3,with_head=True,
            #              BN0_or_LN1=0, input_shape=[1,128,1250],
            #              use_FFN=False, exp_ratio=2, activation='gelu')
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=64, stride=32, padding=32, bias=use_bias),           
            nn.ReLU(),
            Attention(get_nstate(1,attn_dim[1],0),attn_nhead[1],dim=attn_dim[1],dim_num=3)
            #Res_Attention(get_nstate(1,attn_dim[1],0),attn_nhead[1],dim=attn_dim[1],dim_num=3,with_head=True,
            #              BN0_or_LN1=0, input_shape=[1,128,40],
            #              use_FFN=False, exp_ratio=2, activation='gelu')
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=16, stride=4, padding=6, bias=use_bias),           
            nn.ReLU(),
            Attention(get_nstate(2,attn_dim[2],0),attn_nhead[2],dim=attn_dim[2],dim_num=3)
            #Res_Attention(get_nstate(2,attn_dim[2],0),attn_nhead[2],dim=attn_dim[2],dim_num=3,with_head=True,
            #              BN0_or_LN1=0, input_shape=[1,128,10],
            #              use_FFN=False, exp_ratio=2, activation='gelu')
        )
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Sequential(
            nn.Linear(128*10, 128, bias=use_bias),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dense2 = nn.Sequential(
            nn.Linear(128, 128, bias=use_bias),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dense3 = nn.Sequential(
            nn.Linear(128, 128, bias=use_bias),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dense4 = nn.Sequential(
            nn.Linear(128, 128, bias=use_bias),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.out = nn.Linear(128, 128, bias=use_bias)                                      
    
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:       
        x = x.unsqueeze(1)
        # Note that in PyTorch, sequences are in (batch, channels, length)
        # while in TensorFlow, sequences are in (batch, length, channels)
        
        x = self.conv1(x)
        if self.attn_on_dpth:
           x1 = self.flatten1(x).transpose(1,2)     # (B,1,128)
        x = self.conv2(x)
        if self.attn_on_dpth:
           x2 = self.flatten2(x).transpose(1,2)
        x = self.conv3(x)
        if self.attn_on_dpth:
           x3 = self.flatten3(x).transpose(1,2)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        if self.attn_on_dpth:
            d = torch.cat((x1,x2,x3),dim=1) # (B,3,128)
            d = self.LN_AoD(d)      # make sure it's fair for each depth
            d = self.AoD(d)[:,2,:]          # (B,128)
            d = self.pool_AoD(d)            # (B,128)
            x = self.out_AoD(torch.cat((x,d),dim=1))
        else:   
            x = self.dense4(x)        
        x = self.out(x)        
        return x


class net_gram(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        attn_dim_up: list[str] = ['se','se','se','se'],
        attn_nhead_up: list[int] = [1,1,1,1],
        attn_dim_down: list[str] = ['se','se','se','se'],
        attn_nhead_down: list[int] = [1,1,1,1],
        attn_on_dpth: bool = False,
        attn_dim_btw: list[str] = ['none','none','none','none'],
        attn_nhead_btw: list[int] = [1,1,1,1],
        attn_pool: bool = False
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.attn_dim_up = attn_dim_up
        self.attn_nhead_up = attn_nhead_up  
        self.attn_dim_down = attn_dim_down
        self.attn_nhead_down = attn_nhead_down     
        self.attn_on_dpth = attn_on_dpth
        self.attn_dim_btw = attn_dim_btw
        self.attn_nhead_btw = attn_nhead_btw
        self.attn_pool = attn_pool
        
        # !!!
        BN0_or_LN1 = 0
        # !!!

        if attn_on_dpth:
            self.flatten1 = nn.AdaptiveAvgPool2d((4,2))
            self.flatten2 = nn.AdaptiveAvgPool2d((2,2))
            self.flatten3 = nn.AdaptiveAvgPool2d((2,1))
            self.flatten4 = nn.Sequential(nn.BatchNorm2d(128), nn.AdaptiveAvgPool2d((1,1)))
            self.LN_AoD = nn.LayerNorm(128) if BN0_or_LN1 else nn.BatchNorm1d(4)
            self.AoD = Attention(n_state=128, n_head=4, dim='c', dim_num=3)
            #self.pool_AoD = nn.Conv1d(in_channels=128, out_channels=128, group=128, kernel_size=4, padding=0, bias=False)
            self.out_AoD = nn.Sequential(            
                                nn.LayerNorm(256) if BN0_or_LN1 else nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256, 128, bias=use_bias))
        
        # temporal normalization
        self.BN = nn.LayerNorm([311,513]) if BN0_or_LN1 else nn.BatchNorm1d(311)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=use_bias),           
            nn.LayerNorm([156,257]) if BN0_or_LN1 else nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))
        )
        
        self.res1up = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),           
            nn.LayerNorm([77,128]) if BN0_or_LN1 else nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(0,attn_dim_up[0]),attn_nhead_up[0],dim=attn_dim_up[0],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=16)
        )
        # add
        self.res1BN = nn.LayerNorm([77,128]) if BN0_or_LN1 else nn.BatchNorm2d(16)
        self.res1down = nn.Sequential(           
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),           
            nn.ReLU(),
            nn.LayerNorm([77,128]) if BN0_or_LN1 else nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(0,attn_dim_down[0]),attn_nhead_down[0],dim=attn_dim_down[0],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=16)
        )
        # add
        
        self.res2BN1 = nn.Sequential(
            nn.LayerNorm([77,128]) if BN0_or_LN1 else nn.BatchNorm2d(16),
            nn.Identity() if attn_dim_btw[0]=='none' else 
            Res_Attention(get_nstate(0,attn_dim_btw[0]),attn_nhead_btw[0],dim=attn_dim_btw[0],with_head=False,
                          BN0_or_LN1=0, input_shape=[1,16,77,128],
                          use_FFN=False, exp_ratio=2, activation='gelu')
        )
        self.res2up = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=use_bias),           
            nn.LayerNorm([39,64]) if BN0_or_LN1 else nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(1,attn_dim_up[1]),attn_nhead_up[1],dim=attn_dim_up[1],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=32)
        )
        self.res2leak = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,0)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,1), stride=(1,1), padding='same', bias=use_bias)
        )
        # add
        self.res2BN2 = nn.LayerNorm([39,64]) if BN0_or_LN1 else nn.BatchNorm2d(32)
        self.res2down = nn.Sequential(           
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),           
            nn.LayerNorm([39,64]) if BN0_or_LN1 else nn.BatchNorm2d(32),
            nn.ReLU(),            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(1,attn_dim_down[1]),attn_nhead_down[1],dim=attn_dim_down[1],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=32)
        )
        # add
        
        self.res3BN1 = nn.Sequential(
            nn.LayerNorm([39,64]) if BN0_or_LN1 else nn.BatchNorm2d(32),
            nn.Identity() if attn_dim_btw[1]=='none' else 
            Res_Attention(get_nstate(1,attn_dim_btw[1]),attn_nhead_btw[1],dim=attn_dim_btw[1],with_head=False,
                          BN0_or_LN1=0, input_shape=[1,32,39,64],
                          use_FFN=False, exp_ratio=2, activation='gelu')
        )
        self.res3up = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=use_bias),           
            nn.LayerNorm([20,32]) if BN0_or_LN1 else nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(2,attn_dim_up[2]),attn_nhead_up[2],dim=attn_dim_up[2],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=64)
        )
        self.res3leak = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(1,0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), stride=(1,1), padding='same', bias=use_bias)
        )
        # add
        self.res3BN2 = nn.LayerNorm([20,32]) if BN0_or_LN1 else nn.BatchNorm2d(64)
        self.res3down = nn.Sequential(           
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),           
            nn.LayerNorm([20,32]) if BN0_or_LN1 else nn.BatchNorm2d(64),
            nn.ReLU(),            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(2,attn_dim_down[2]),attn_nhead_down[2],dim=attn_dim_down[2],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=64)
        )
        # add
        
        self.res4BN1 = nn.Sequential(
            nn.LayerNorm([20,32]) if BN0_or_LN1 else nn.BatchNorm2d(64),
            nn.Identity() if attn_dim_btw[2]=='none' else 
            Res_Attention(get_nstate(2,attn_dim_btw[2]),attn_nhead_btw[2],dim=attn_dim_btw[2],with_head=False,
                          BN0_or_LN1=0, input_shape=[1,64,20,32],
                          use_FFN=False, exp_ratio=4, activation='gelu')
        )
        self.res4up = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=use_bias),           
            nn.LayerNorm([10,16]) if BN0_or_LN1 else nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(3,attn_dim_up[3]),attn_nhead_up[3],dim=attn_dim_up[3],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=128)
        )
        self.res4leak = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), stride=(1,1), padding='same', bias=use_bias)
        )
        # add
        self.res4BN2 = nn.LayerNorm([10,16]) if BN0_or_LN1 else nn.BatchNorm2d(128)
        self.res4down = nn.Sequential(           
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),           
            nn.LayerNorm([10,16]) if BN0_or_LN1 else nn.BatchNorm2d(128),
            nn.ReLU(),            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding='same', bias=use_bias),
            Attention(get_nstate(3,attn_dim_down[3]),attn_nhead_down[3],dim=attn_dim_down[3],with_head=False)
            #SqueezeAndExcitationBlock(num_channels=128)
        )
        # add
        
        no_post_attn = (attn_dim_btw[3]=='none') and (attn_dim_btw[4]=='none') and (attn_dim_btw[5]=='none') and (not attn_pool)
        self.pool = nn.Sequential(
            nn.Identity() if no_post_attn else nn.LayerNorm([10,16]) if BN0_or_LN1 else nn.BatchNorm2d(128),
            nn.Identity() if attn_dim_btw[3]=='none' else 
            Res_Attention(get_nstate(3,attn_dim_btw[3]),attn_nhead_btw[3],dim=attn_dim_btw[3],with_head=False,
                          BN0_or_LN1=BN0_or_LN1, #norm_dim=[10,16] if BN0_or_LN1 else 128,
                          use_FFN=False, exp_ratio=4, activation='gelu'),
            nn.Identity() if attn_dim_btw[4]=='none' else 
            Res_Attention(get_nstate(3,attn_dim_btw[4]),attn_nhead_btw[4],dim=attn_dim_btw[4],with_head=False,
                          BN0_or_LN1=BN0_or_LN1, #norm_dim=[10,16] if BN0_or_LN1 else 128,
                          use_FFN=False, exp_ratio=4, activation='gelu'),
            nn.Identity() if attn_dim_btw[5]=='none' else 
            Res_Attention(get_nstate(3,attn_dim_btw[5]),attn_nhead_btw[5],dim=attn_dim_btw[5],with_head=False,
                          BN0_or_LN1=BN0_or_LN1, #norm_dim=[10,16] if BN0_or_LN1 else 128,
                          use_FFN=False, exp_ratio=4, activation='gelu'),
            nn.MaxPool2d(kernel_size=(10,1), stride=(1,1), padding=(0,0)) if not attn_pool else
            Attention(get_nstate(3,'t'),1,dim='t') # Note that it would be repetitive if not attn_dim_btw[2]=='none')
        )
                    
        self.out = nn.Sequential(nn.Flatten(),
            nn.LayerNorm(128*16) if BN0_or_LN1 else nn.BatchNorm1d(128*16),
            nn.Linear(128*16, 128, bias=use_bias)
        )             
        
       
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True) # temporal normalization
        x = self.BN(x).unsqueeze(1)
        # Note that in PyTorch, images are in (batch, channels, height, width)
        # while in TensorFlow, images are in (batch, height, width, channels)
        
        x = self.conv(x)
        
        xr = self.res1up(x)
        x = x + xr
        x = self.res1BN(x)
        xr = self.res1down(x)
        x = x + xr
        
        x = self.res2BN1(x)
        if self.attn_on_dpth:
           res1 = self.flatten1(x).flatten(start_dim=1).unsqueeze(1)        
        xr = self.res2up(x)
        x = self.res2leak(x)
        x = x + xr
        x = self.res2BN2(x)
        xr = self.res2down(x)
        x = x + xr
        
        x = self.res3BN1(x)
        if self.attn_on_dpth:
           res2 = self.flatten2(x).flatten(start_dim=1).unsqueeze(1)
        xr = self.res3up(x)
        x = self.res3leak(x)
        x = x + xr
        x = self.res3BN2(x)
        xr = self.res3down(x)
        x = x + xr
        
        x = self.res4BN1(x)
        if self.attn_on_dpth:
           res3 = self.flatten3(x).flatten(start_dim=1).unsqueeze(1)
        xr = self.res4up(x)
        x = self.res4leak(x)
        x = x + xr
        x = self.res4BN2(x)
        xr = self.res4down(x)
        x = x + xr
        if self.attn_on_dpth:
           res4 = self.flatten4(x).flatten(start_dim=1).unsqueeze(1)
                   
        x = self.pool(x)
        if self.attn_pool:
            x = x[:,:,-1,:]
        x = self.out(x)        
        if not self.attn_on_dpth:
            return x
        else:
            d = torch.cat((res1,res2,res3,res4),dim=1)  # (B,4,128)
            d = self.LN_AoD(d)      # make sure it's fair for each depth
            d = self.AoD(d)[:,3,:]      # (B,128)
            #d = self.pool_AoD(d.transpose(1,2)).squeeze(2)  # (B,128,1) -> (B,128)
            return self.out_AoD(torch.cat((x,d),dim=1))


class Dual_Branch(nn.Module):
    def __init__(
        self,
        loss: nn.Module,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        spec_aug: bool = False,
        spec_aug_conf: Dict[str, Any] = {}
    ) -> None:
    
        super().__init__()
        self.loss = loss
        self.embedding_dim = embedding_dim
        self.l2_weight_decay = l2_weight_decay
        self.use_bias = use_bias
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        
        self.net_trum = net_trum(embedding_dim, l2_weight_decay, use_bias)
        self.net_gram = net_gram(embedding_dim, l2_weight_decay, use_bias)
        if spec_aug:
            self.net_SpecAug = SpecAug(spec_aug_conf)   # input specgram (Tensor): Tensor of shape `(..., freq, time)
        
        self.register_buffer("window_stft", torch.hann_window(1024) ) 
        self.register_buffer("window_fft", torch.hann_window(160000).unsqueeze(0) )
            
    def process_audio(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:    
        if wav.shape[1] > 160000:
            wav = wav[:,:160000]
        elif wav.shape[1] < 160000:
            wav = F.pad(wav, (0, 160000-wav.shape[1]), mode='constant', value=0)
              
        stft = torch.stft(wav, 1024, 512, window=self.window_stft, center=False, onesided=True, return_complex=True).abs().to(dtype=wav.dtype)
        # Note that torch.stft() output (B?,N_freq,N_frame)
        if self.spec_aug:
            stft = self.net_SpecAug(stft)
        stft = stft.permute(0,2,1)    
        
        fft = torch.fft.fft( wav * self.window_fft )
        fft = fft[:,:80000].abs().to(dtype=wav.dtype)
        
        return fft, stft
        
    def forward(
        self,
        wav: torch.Tensor,
        label: torch.Tensor = None,      
        out_emb: bool = False        
    ) -> Dict[str, Any]:
        
        trum, gram = self.process_audio(wav)        
        trum = self.net_trum(trum)
        gram = self.net_gram(gram)
        emb = torch.cat((gram,trum), dim=1)
        
        if out_emb:
            #output_dict = {'embedding':emb, 'branch1':trum, 'branch2':gram}           
            output_dict = {'embedding':emb}
        else:
            output_dict = self.loss(emb, label)                               
        return output_dict

class Dual_Branch_multi(nn.Module):
    def __init__(
        self,
        loss0: nn.Module,
        loss1: nn.Module,
        loss2: nn.Module,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        spec_aug: bool = False,
        spec_aug_conf: Dict[str, Any] = {}
    ) -> None:
    
        super().__init__()
        self.loss0 = loss0
        self.loss1 = loss1
        self.loss2 = loss2
        self.embedding_dim = embedding_dim
        self.l2_weight_decay = l2_weight_decay
        self.use_bias = use_bias
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        
        self.net_trum = net_trum(embedding_dim, l2_weight_decay, use_bias)
        self.net_gram = net_gram(embedding_dim, l2_weight_decay, use_bias)
        if spec_aug:
            self.net_SpecAug = SpecAug(spec_aug_conf)   # input specgram (Tensor): Tensor of shape `(..., freq, time)
        
        self.register_buffer("window_stft", torch.hann_window(1024) ) 
        self.register_buffer("window_fft", torch.hann_window(160000).unsqueeze(0) )
            
    def process_audio(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:    
        if wav.shape[1] > 160000:
            wav = wav[:,:160000]
        elif wav.shape[1] < 160000:
            wav = F.pad(wav, (0, 160000-wav.shape[1]), mode='constant', value=0)
              
        stft = torch.stft(wav, 1024, 512, window=self.window_stft, center=False, onesided=True, return_complex=True).abs().to(dtype=wav.dtype)
        # Note that torch.stft() output (B?,N_freq,N_frame) 
        if self.spec_aug:
            stft = self.net_SpecAug(stft)
        stft = stft.permute(0,2,1)              
        
        fft = torch.fft.fft( wav * self.window_fft )
        fft = fft[:,:80000].abs().to(dtype=wav.dtype)

        return fft, stft
        
    def forward(
        self,
        wav: torch.Tensor,
        label: torch.Tensor = None,      
        out_emb: bool = False
    ) -> Dict[str, Any]:
        
        trum, gram = self.process_audio(wav)        
        trum = self.net_trum(trum)
        gram = self.net_gram(gram)
        emb = torch.cat((gram,trum), dim=1)
        
        if out_emb:
            #output_dict = {'embedding':emb, 'branch1':trum, 'branch2':gram}           
            output_dict = {'embedding':emb}
            return output_dict
        else:
            l0 = self.loss0(emb, label)
            l1 = self.loss1(gram,label)
            l2 = self.loss2(trum,label)                      
            return l0, l1, l2


# > in order to independently employ the 2 branch models (in MultiBranch)
        
class Dual_Branch_spectrum(nn.Module):    # work as second branch in Kevin's original work
    def __init__(
        self,
        loss: nn.Module,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        attn_dim: list[str] = ['se','se','se'],
        attn_nhead: list[int] = [4,4,4],
        attn_on_dpth: bool = False
    ) -> None:       
        super().__init__()
        self.loss = loss
        self.embedding_dim = embedding_dim
        self.l2_weight_decay = l2_weight_decay
        self.use_bias = use_bias
        
        self.attn_dim = attn_dim
        self.attn_nhead = attn_nhead
        self.attn_on_dpth = attn_on_dpth       
        
        self.net_trum = net_trum(embedding_dim, l2_weight_decay, use_bias,
                                 attn_dim, attn_nhead, attn_on_dpth)        
        self.register_buffer("window_fft", torch.hann_window(160000).unsqueeze(0) )
            
    def process_audio(
        self,
        wav: torch.Tensor,
    ) -> torch.Tensor:    
        if wav.shape[1] > 160000:
            wav = wav[:,:160000]
        elif wav.shape[1] < 160000:
            wav = F.pad(wav, (0, 160000-wav.shape[1]), mode='constant', value=0)                
        
        fft = torch.fft.fft( wav * self.window_fft )
        fft = fft[:,:80000].abs().to(dtype=wav.dtype)
        
        return fft
        
    def forward(
        self,
        wav: torch.Tensor,
        label: torch.Tensor = None,      
        out_emb: bool = False        
    ) -> Dict[str, Any]:
        
        trum = self.process_audio(wav)        
        emb = self.net_trum(trum)
        
        if out_emb:
            #output_dict = {'embedding':emb, 'branch1':trum, 'branch2':gram}           
            output_dict = {'embedding':emb}
        else:
            output_dict = self.loss(emb, label)                               
        return output_dict
    
    
class Dual_Branch_spectrogram(nn.Module):     # work as first branch in Kevin's original work
    def __init__(
        self,
        loss: nn.Module,
        embedding_dim: int = 128,
        l2_weight_decay: float = 1e-5,
        use_bias: bool = False,
        spec_aug: bool = False,
        spec_aug_conf: Dict[str, Any] = {},
        attn_dim_up: list[str] = ['se','se','se','se'],
        attn_nhead_up: list[int] = [1,1,1,1],
        attn_dim_down: list[str] = ['se','se','se','se'],
        attn_nhead_down: list[int] = [1,1,1,1],
        attn_on_dpth: bool = False,
        attn_dim_btw: list[str] = ['none','none','none','none'],
        attn_nhead_btw: list[int] = [1,1,1,1],
        attn_pool: bool = False
    ) -> None:       
        super().__init__()
        self.loss = loss
        self.embedding_dim = embedding_dim
        self.l2_weight_decay = l2_weight_decay
        self.use_bias = use_bias
        self.spec_aug = spec_aug
        self.spec_aug_conf = spec_aug_conf
        
        self.attn_dim_up = attn_dim_up
        self.attn_nhead_up = attn_nhead_up  
        self.attn_dim_down = attn_dim_down
        self.attn_nhead_down = attn_nhead_down     
        self.attn_on_dpth = attn_on_dpth 
        self.attn_dim_btw = attn_dim_btw
        self.attn_nhead_btw = attn_nhead_btw
        self.attn_pool = attn_pool
              
        self.net_gram = net_gram(embedding_dim, l2_weight_decay, use_bias,
                                 attn_dim_up, attn_nhead_up, attn_dim_down, attn_nhead_down, attn_on_dpth,
                                 attn_dim_btw,attn_nhead_btw,attn_pool)
        if spec_aug:
            self.net_SpecAug = SpecAug(spec_aug_conf)   # input specgram (Tensor): Tensor of shape `(..., freq, time)       
        self.register_buffer("window_stft", torch.hann_window(1024) ) 
            
    def process_audio(
        self,
        wav: torch.Tensor,
        out_emb: bool = False
    ) -> torch.Tensor:    
        if wav.shape[1] > 160000:
            wav = wav[:,:160000]
        elif wav.shape[1] < 160000:
            wav = F.pad(wav, (0, 160000-wav.shape[1]), mode='constant', value=0)
              
        stft = torch.stft(wav, 1024, 512, window=self.window_stft, center=False, onesided=True, return_complex=True).abs().to(dtype=wav.dtype)
        # Note that torch.stft() output (B?,N_freq,N_frame)
        if self.spec_aug and (not out_emb):
            stft = self.net_SpecAug(stft)
        stft = stft.permute(0,2,1)    
        
        return stft
        
    def forward(
        self,
        wav: torch.Tensor,
        label: torch.Tensor = None,      
        out_emb: bool = False        
    ) -> Dict[str, Any]:
        
        gram = self.process_audio(wav, out_emb=out_emb)        
        emb = self.net_gram(gram)
        
        if out_emb:
            #output_dict = {'embedding':emb, 'branch1':trum, 'branch2':gram}           
            output_dict = {'embedding':emb}
        else:
            output_dict = self.loss(emb, label)                               
        return output_dict