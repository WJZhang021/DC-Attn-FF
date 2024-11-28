import os

from typing import Dict, Any, Optional
from torch import nn

from models.loss.loss_wrapper import construct_loss


def construct_model(
    args: Dict[str, Any],
    with_loss: bool = True
):
    if with_loss:
        loss_conf = args.get('loss_conf', {})
        loss_conf['num_features'] = args['emb_size']
        loss_conf['num_classes'] = args['num_classes']
        loss = construct_loss(
            loss_name=args.get('loss_name', 'arcface'),
            **loss_conf
        )
        
        if 'multi_branch' in args:            
            if args['multi_branch'] == True:               
                multi_branch_conf = args.get('multi_branch_conf', {})                                              
                branch_emb_size_list = multi_branch_conf['branch_emb_size']

                branch_loss_name_list = multi_branch_conf.get('branch_loss_name', [])
                if branch_loss_name_list==[]:
                    branch_loss_name = args.get('loss_name', 'arcface')
                    branch_loss_name_list = [branch_loss_name] * args['multi_branch_num']
                
                loss_list = [loss] 
                for i in range(args['multi_branch_num']):
                    loss_conf['num_features'] = branch_emb_size_list[i]
                    loss = construct_loss(
                        loss_name=branch_loss_name_list[i],
                        **loss_conf
                    )
                    loss_list.append(loss)
                # for multi_BP: False, this loss_list would still be constructed, as construction of a branch model require a loss
                # but branch loss would not be actually used in training process (unless flexible_allocate: True)
    else:
        loss = None

    model_conf = args.get('model_conf', {})
    if 'ckpt' in model_conf.keys():
        if not os.path.exists(model_conf['ckpt']):
            model_conf['ckpt'] = os.path.join(args['model_dir'], model_conf['ckpt'])

    if args['model_name'] in ['wav2vec_300m',
                              'wav2vec_1b',
                              'wav2vec_2b',
                              'hubert_base',
                              'hubert_large',
                              'hubert_xlarge',
                              'unispeech_base',
                              'unispeech_large',
                              'wavlm_base',
                              'wavlm_large',
                              'data2vec']:
        from models.w2v.w2v import W2V
        net = W2V(
            loss=loss,
            model_name=args['model_name'],
            embedding_dim=args['emb_size'],
            freeze=args['freeze'],
            **model_conf
        )

    elif args['model_name'] == 'audiomae':
        from models.audiomae.audiomae import AudioMAE
        net = AudioMAE(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'ast':
        from models.ast.ast import AST
        net = AST(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'ssast':
        from models.ssast.ssast import SSAST
        net = SSAST(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'imagebind':
        from models.imagebind.imagebind_audio import ImageBind
        net = ImageBind(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'qwen_audio':
        from models.qwen_audio.qwen_audio import Qwen_Audio
        net = Qwen_Audio(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'w2v_bert_2':
        from models.w2v_bert_2.w2v_bert_2 import W2V_BERT_2
        net = W2V_BERT_2(
            loss=loss,
            embedding_dim=args['emb_size'],
        )

    elif args['model_name'] == 'mobilenetv2':
        from models.mobilenetv2.mobilenetv2 import MobileNetV2
        net = MobileNetV2(
            loss=loss,
            embedding_dim=args['emb_size'],
        )

    elif args['model_name'] == 'mobilefacenet':
        from models.mfn.MobileFaceNet import MobileFacenet
        fbank_conf = args['fbank_conf']
        sr = args['sample_rate']
        input_points = int(args['input_duration'] * sr)
        n_fft = int(fbank_conf.get('frame_length', 25) * sr // 1000)
        win_shift = int(fbank_conf.get('frame_shift', 10) * sr // 1000)
        tframe = (input_points - n_fft) // win_shift + 1
        net = MobileFacenet(
            loss=loss,
            embedding_dim=args['emb_size'],
            freq_bins=fbank_conf.get('num_mel_bins', 80),
            tframe=tframe
        )

    elif args['model_name'] == 'eat':
        from models.eat.EAT import EAT
        net = EAT(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'eat_lora':
        from models.eat_lora.EAT import EAT
        net = EAT(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'beats':
        from models.beats.beats_ft import BEATs_FT
        net = BEATs_FT(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'beats_lora':
        from models.beats_lora.beats_ft import BEATs_FT
        net = BEATs_FT(
            loss=loss,
            embedding_dim=args['emb_size'],
            Adalora=args['Adalora'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'atst':
        from models.atst.atst import ATST
        net = ATST(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'cnn10':
        from models.panns.cnn10 import CNN10
        net = CNN10(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'resnet':
        from models.resnet.resnet import ResNet34
        net = ResNet34(
            loss=loss,
            embed_dim=args['emb_size'],
            **args.get('model_conf', {})
        )

    elif args['model_name'] == 'ecapa_tdnn':
        from models.tdnn.ecapa_tdnn import ECAPA_TDNN_c512
        net = ECAPA_TDNN_c512(
            loss=loss,
            embed_dim=args['emb_size'],
            **args.get('model_conf', {})
        )
        
    elif args['model_name'] == 'ced':
        from models.ced.ced import CED
        net = CED(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )
        
    elif args['model_name'] == 'conveat':
        from models.conveat.ConvEAT import ConvEAT
        net = ConvEAT(
            loss=loss,
            embedding_dim=args['emb_size'],
            **args.get('model_conf', {})
        )
    
    elif args['model_name'] == 'whisper':
        from models.whisper.whisper import WhisperEncoder
        net = WhisperEncoder(
            loss=loss,
            embedding_dim=args['emb_size'],
            sec=args['input_duration'],
            pooling_head=args['pooling_head'],
            **args.get('model_conf', {})
        )
    elif args['model_name'] in ['Kevin',
                              'dual_branch']:
        from models.dual_branch.dual_branch import Dual_Branch
        net = Dual_Branch(
            loss=loss,
            **args.get('model_conf', {}),
            spec_aug = args['spec_aug'],
            spec_aug_conf = args.get('spec_aug_conf', {})
        )
    elif args['model_name'] in ['Kevin_multi',
                              'dual_branch_multi']:
        from models.dual_branch.dual_branch import Dual_Branch_multi
        net = Dual_Branch_multi(
            *loss_list,
            **args.get('model_conf', {}),
            spec_aug = args['spec_aug'],
            spec_aug_conf = args.get('spec_aug_conf', {})
        )
               
    elif args['model_name'] == 'multibranch':
        from models.multibranch import MultiBranch
        multi_branch_conf = args.get('multi_branch_conf', [])
        multi_branch_conf.pop('branch_loss_name', None)
        multi_branch_conf.pop('branch_loss_weight', None)
       
        spec_aug = args['spec_aug'],
        spec_aug_conf = args.get('spec_aug_conf', {})
        
        if multi_branch_conf['model_list'][0] == 'Kevin_spectrogram':
            multi_branch_conf['model_conf_list'][0]['spec_aug'] = spec_aug
            multi_branch_conf['model_conf_list'][0]['spec_aug_conf'] = spec_aug_conf
        
        multi_input = (args['feat_type']=='both') or args.get('multi_fbank', False)
        net = MultiBranch(
            multi_branch_num = args['multi_branch_num'],
            total_emb_size = args['emb_size'],
            loss_list = nn.ModuleList(loss_list).cuda(),
            multi_input = multi_input,
            **multi_branch_conf          
        )

    else:
        raise NotImplementedError(f"Model {args['model_name']} is not implemented!")

    return net
