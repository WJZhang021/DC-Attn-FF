from typing import Dict, Any, Iterable, Optional
import os

import numpy as np
import torch
from torch import nn

def get_medata(medata: Dict[str,torch.Tensor], args: Dict[str,Any]):
    feat_type = args.get('feat_type', 'wav')
    multi_fbank = args.get('multi_fbank', False)
    if feat_type == 'both':
        bs =  medata['wav'].shape[0]
        if not multi_fbank:
            x_out = (medata['wav'].cuda(), medata['fbank'].cuda())
        else:
            x_out = tuple((medata['wav'].cuda(),) + tuple(x.cuda() for x in medata['fbank']))
    elif feat_type == 'fbank':        
        if not multi_fbank:
            bs =  medata['fbank'].shape[0]
            x_out = medata['fbank'].cuda()
        else:
            bs =  medata['fbank'][0].shape[0]
            x_out = tuple((None) + tuple(x.cuda() for x in medata['fbank']))
    else:
        bs =  medata['wav'].shape[0]
        x_out = medata['wav'].cuda()
    return bs, x_out

def net_forward(net: nn.Module, medata: Dict[str,torch.Tensor], args: Dict[str,Any],\
    train0_test1: bool = False):
    bs, x = get_medata(medata, args)
    need_actual_len = args.get('need_actual_len', False)
    if not train0_test1:    
        label = medata['classify_labs'].cuda()        
        if need_actual_len:
            len = medata['valid_len'].cuda()
            output_dict = net(x, label, valid_len=len, out_emb=False)
        else:
            output_dict = net(x, label, out_emb=False)
    else:
        if need_actual_len:
            len = medata['valid_len'].cuda()
            output_dict = net(x, valid_len=len, out_emb=True)
        else:
            output_dict = net(x, out_emb=True)    
    return bs, output_dict

def loss_backward(output_dict: Dict[str,Any], args: Dict[str,Any]):
    skip_flag = 0
    multi_branch = args.get('multi_branch', False)            
    if multi_branch:               
        multi_branch_conf = args.get('multi_branch_conf', {})
        if multi_branch_conf['multi_BP']:
            branch_loss_weight_list = multi_branch_conf.get('branch_loss_weight', [])
            if branch_loss_weight_list==[]:
                branch_loss_weight_list = [1.0] * args['multi_branch_num']
            
            for i in range(args['multi_branch_num']):                       
                loss_branch = output_dict[i+1]['loss'] * branch_loss_weight_list[i]           
                loss_branch.backward(retain_graph=True)
            output_dict = output_dict[0]  
            
        elif multi_branch_conf['indp_BP']:
            skip_flag = 1
            branch_loss_weight_list = multi_branch_conf.get('branch_loss_weight', [])
            if branch_loss_weight_list==[]:
                branch_loss_weight_list = [1.0] * args['multi_branch_num']
                                
            for i in range(args['multi_branch_num']):                        
                loss_branch = output_dict[i+1]['loss'] * branch_loss_weight_list[i]           
                loss_branch.backward(retain_graph=True)                                       
            [loss_item, acc_num] = np.mean([[branch['loss'].item(),branch['acc_num'].cpu().item()] for branch in output_dict[1:]], axis=0)                  
            # Note that the following processes about performance,
            # especially lr_sch.step(avg_loss), are based on the average of each branch
    if not skip_flag:
        loss = output_dict['loss']
        loss.backward()            
        loss_item = loss.item()
        acc_num = output_dict['acc_num'].cpu().item()
    return loss_item, acc_num

def save_model(net: nn.Module, args: Dict[str,Any], model_path: str, model_name: str):
    mn = args.get('model_name', '')
    if (mn == 'MultiBranch' or mn == 'multibranch') and args.get('separate_save', False):               
        branch_list = args['multi_branch_conf']['model_list']
        for i in range(args['multi_branch_num']):
            branch_name = branch_list[i] +"-"+ model_name                        
            torch.save(net.models[i].state_dict(), os.path.join(model_path, branch_name))                    
    else:
        torch.save(net.state_dict(), os.path.join(model_path, model_name))
    return None