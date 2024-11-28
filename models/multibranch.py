import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

class DynamicLinear(nn.Module):
    def __init__(self, input_dim: int, max_out_dim: int, initial_out_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("branch_emb_size must be positive number")
        self.input_dim = input_dim
        self.max_out_dim = max_out_dim
        self.current_out_dim = initial_out_dim
   
        self.pre = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
        )       
        # initialize        
        w1 = torch.rand(input_dim, max_out_dim).to(dtype=torch.float)
        w1 = (2.0 * w1 - 1.0) / (input_dim ** 0.5)
        self.weights1 = nn.Parameter(w1)
        self.bias1 = nn.Parameter(torch.zeros(max_out_dim))
    
    def set_out_dim(self, dim: int)-> None:
        self.current_out_dim = dim
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        if self.current_out_dim > self.max_out_dim:
            raise ValueError("DynamicLinear: current_out_dim exceeds max_out_dim")
        if not x.shape[1] == self.input_dim:
            print(f"! WARNING !: branch_emb_size turns out to be {x.shape[1]} rather than {self.input_dim}")
        x = self.pre(x)        
        weight = self.weights1[:x.shape[1], :self.current_out_dim]
        bias = self.bias1[:self.current_out_dim]
        x = torch.addmm(bias, x, weight)

        return x


class MultiBranch(nn.Module):
    def __init__(
        self,
        loss_list: nn.ModuleList,     # loss_list[0] is the major loss for the total embedding
        multi_branch_num: int,
        model_list: list[str] = [],
        model_conf_list: list[Dict[str, Any]] = [],
        multi_input: bool = False,
        input_type_list: list[int] = [],
        total_emb_size: int = 128,
        branch_emb_size: list[int] = [],
        multi_BP: bool = True,
        indp_BP: bool = True,
        branch_emb_proj: bool = False,
        branch_emb_size_after_proj: list[int] = [],
        flexible_allocate: bool = False,
        flexible_allocate_interval: int = 200        
    ) -> None:
    
        super(MultiBranch, self).__init__()
        self.loss_list = loss_list
        self.multi_branch_num = multi_branch_num
        self.model_name_list = model_list
        self.model_conf_list = model_conf_list
        self.multi_input = multi_input
        self.input_type_list = input_type_list
        self.total_emb_size = total_emb_size
        self.branch_emb_size = branch_emb_size
        self.multi_BP = multi_BP
        self.indp_BP = indp_BP
        self.branch_emb_proj = branch_emb_proj
        self.branch_emb_size_after_proj = torch.tensor(branch_emb_size_after_proj, dtype=torch.int)
        self.flexible_allocate = flexible_allocate
        self.flexible_allocate_interval = flexible_allocate_interval

        if self.multi_BP and self.indp_BP:
            print('! WARNING !: multi_BP and indp_BP are 2 conflicting modes')
            print('             presume as multi_BP')
            self.indp_BP = False
        
        if self.flexible_allocate and not self.branch_emb_proj:
            print('! WARNING !: flexible_allocate: True while branch_emb_proj: False')
            print('             has been automatically changed to branch_emb_proj: True')
            self.branch_emb_proj = True
        '''
        if self.flexible_allocate and not self.multi_BP and not self.indp_BP:
            print('! WARNING !: flexible_allocate: True while not multi_BP & not indp_BP')
            print('             will cause ERROR')
            self.multi_BP = True
        '''
        if not branch_emb_proj:
            SUPPOESED_total_emb_size = sum(branch_emb_size)
        else:
            SUPPOESED_total_emb_size = sum(branch_emb_size_after_proj)
        if not total_emb_size == SUPPOESED_total_emb_size:
            print('! WARNING !: total_emb_size is not appropriate')
            print('             has been automatically changed from',total_emb_size,'to',SUPPOESED_total_emb_size)
            self.total_emb_size = SUPPOESED_total_emb_size
        
        if self.flexible_allocate:
            self.branch_emb_size_after_proj_r = self.branch_emb_size_after_proj / self.total_emb_size      
        
           
        models = []
        for i in range(multi_branch_num):
            model_name = self.model_name_list[i]
            model_conf = self.model_conf_list[i]            
                        
            emb_size = self.branch_emb_size[i]            
            loss = self.loss_list[i+1]
            
            if model_name == 'whisper':
                from models.whisper.whisper import WhisperEncoder
                net = WhisperEncoder(
                    loss=loss,
                    embedding_dim=emb_size,
                    **model_conf
                )   # Note to include sec & pooling_head in model_conf !
            
            elif model_name in ['Kevin',
                              'dual_branch']:
                from models.dual_branch.dual_branch import Dual_Branch
                net = Dual_Branch(
                    *loss_list,
                    **model_conf
                )   # Note to include spec_aug & spec_aug_conf in model_conf !
            elif model_name in ['Kevin_spectrum',
                              'dual_branch_spectrum']:
                from models.dual_branch.dual_branch import Dual_Branch_spectrum
                net = Dual_Branch_spectrum(
                    loss=loss,
                    **model_conf
                )
            elif model_name in ['Kevin_spectrogram',
                              'dual_branch_spectrogram']:
                from models.dual_branch.dual_branch import Dual_Branch_spectrogram
                net = Dual_Branch_spectrogram(
                    loss=loss,
                    **model_conf
                )   # Note to include spec_aug & spec_aug_conf in model_conf !                      
            
            
            elif model_name == 'eat':
                from models.eat.EAT import EAT
                net = EAT(
                    loss=loss,
                    embedding_dim=emb_size,
                    **model_conf
                )
            elif model_name == 'beats':
                from models.beats.beats_ft import BEATs_FT
                net = BEATs_FT(
                    loss=loss,
                    embedding_dim=emb_size,
                    **model_conf
                )
            elif model_name == 'beats_lora':
                from models.beats_lora.beats_ft import BEATs_FT
                net = BEATs_FT(
                    loss=loss,
                    embedding_dim=emb_size,
                    **model_conf
                )
            
            else:
                raise NotImplementedError(f"Model {model_name} is not implemented!")            
            
            models.append(net)
        self.models = nn.ModuleList(models)
            
        if self.branch_emb_proj:
            projs = []
            for i in range(multi_branch_num):
                emb_size = self.branch_emb_size[i]         
                emb_size_after_proj = int(self.branch_emb_size_after_proj[i])
                
                if self.flexible_allocate:
                    proj = DynamicLinear(emb_size, emb_size*2, emb_size_after_proj)
                else:
                    proj = nn.Linear(emb_size, emb_size_after_proj)
                projs.append(proj)
            self.projs = nn.ModuleList(projs)  
            
            if self.flexible_allocate:
                self.counter = 0  
        
    def re_allocate(
        self,
        loss: torch.Tensor,     # loss[0] should be the major loss for the total embedding
        alpha: float = 20,      # how strongly loss_ratio result in contribution difference
        beta: float = 1.0,      # how aggressively allocation_ratio is updated
        toprint: bool = False
    ) -> None:
        if not self.flexible_allocate:
            return None                            
        if self.indp_BP:
            loss[0] = torch.mean(loss[1:])
        contri_r = torch.log( alpha * loss[0]/loss[1:] + 1.0) / self.branch_emb_size_after_proj_r
        allocate_r = contri_r / torch.sum(contri_r)

        self.branch_emb_size_after_proj_r = ( beta*allocate_r + self.branch_emb_size_after_proj_r) / (beta+1.0)
        self.branch_emb_size_after_proj = torch.round(self.branch_emb_size_after_proj_r * self.total_emb_size).to(torch.int)            
        res = self.total_emb_size - torch.sum(self.branch_emb_size_after_proj)
        self.branch_emb_size_after_proj[-1] = torch.round(self.branch_emb_size_after_proj[-1] + res).to(torch.int)  
        for i in range(self.multi_branch_num):
            self.projs[i].set_out_dim(self.branch_emb_size_after_proj[i].item()) 
        
        if toprint:
            print('branch_emb_size re-allocate:',[emb.cpu().item() for emb in self.branch_emb_size_after_proj])
        return None              
    
    def forward(
        self,
        input: torch.Tensor = None,     # could be wav / fbank / (wav,fbank,) 
        label: torch.Tensor = None,      
        out_emb: bool = False
    ) -> Dict[str, Any]:                               
        flag_reallocate = 0
        if (not out_emb) and self.flexible_allocate:
            self.counter += 1        
            if self.counter >= self.flexible_allocate_interval:
                self.counter -= self.flexible_allocate_interval
                flag_reallocate = 1
        
        if self.multi_input:
            bs = input[1].shape[0]
            dtype = input[1].dtype
            device = input[1].device
        else:
            bs = input.shape[0]
            dtype = input.dtype
            device = input.device
        total_emb = torch.zeros([bs,0]).to(dtype=dtype, device=device)
        loss_output_list = [0]      # loss_output_list[0] should be the major loss for the total embedding 
        
        for i in range(self.multi_branch_num):
            if self.multi_input:
                input_type = self.input_type_list[i]                
                branch_emb = self.models[i](input[input_type], label, out_emb=True)['embedding']
            else:
                branch_emb = self.models[i](input, label, out_emb=True)['embedding']
                       
            
            if flag_reallocate or ((not out_emb) and (self.multi_BP or self.indp_BP)):      
                loss_output = self.loss_list[i+1](branch_emb, label)   # to use self.models[i].loss is the same
                loss_output_list.append(loss_output) 
            
            if self.branch_emb_proj:
                branch_emb = self.projs[i](branch_emb)                               
            total_emb = torch.cat((total_emb,branch_emb), dim=1)
                   
        if flag_reallocate or ((not out_emb) and self.multi_BP):
            loss_output_list[0] = self.loss_list[0](total_emb, label)
        
        if flag_reallocate:
            loss_output_list_item = [0]
            for i in range(self.multi_branch_num):
                loss_output_list_item.append(loss_output_list[i+1]['loss'].item())
            if not self.indp_BP:
                loss_output_list_item[0] = loss_output_list[0]['loss'].item()            
            self.re_allocate(torch.tensor(loss_output_list_item), toprint=True)
                
        if out_emb:           
            output_dict = {'embedding':total_emb}
            return output_dict
        elif self.multi_BP or self.indp_BP:                                          
            return loss_output_list
        else:
            return self.loss_list[0](total_emb, label)