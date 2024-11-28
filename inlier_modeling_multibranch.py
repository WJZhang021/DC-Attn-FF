import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'  # TODO:

import gc
import copy
import yaml
import glob
import torch
import logging
import pickle
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Dict, Tuple, Any
from scipy.stats import hmean
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from models.model_wrapper import construct_model
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from util import (
    FixedSizeQueue,
    set_seed,
    ensemble_embs,
    stepDataloader,
    find_best_ckpt_emb,
)
from datasets.audio_dataset import (
    build_dcase_dataset,
    build_idmt_engine_dataset
)
from datasets.label_func import (
    DCASE_Machine_Labeler,
    DCASE_Section_Labeler,
    DCASE_Machine_Section_Labeler,
    DCASE_Attribute_Labeler,
    DCASE_Pseudo_Labeler,
    IDMT_Engine_Labeler
)
from uni_detect.task.dcase.infer_score import DCASE_Anomaly_Detection
from uni_detect.task.dcase.cal_aucs import cal_aucs

from util_multibranch import get_medata,net_forward,loss_backward,save_model

def train_and_test(
    net: nn.Module,
    train_ds: Dataset,
    test_ds: Dict[str, Dataset],
    args: Dict,
    writer: SummaryWriter
) -> None:
    best_score = 0.0
    best_it = 0
    
    model_path = os.path.join(args['exp_dir'], "saved_models")
    os.makedirs(model_path, exist_ok=True)

    learning_curves = []
    use_lora = args.get('lora', False)
    if use_lora:
        try:
            import loralib as lora
            lora.mark_only_lora_as_trainable(net)
            lora_path = os.path.join(model_path, "lora")
            os.makedirs(lora_path, exist_ok=True)
        except Exception:
            raise Warning('lora module is not found. switch to full fine-tuning')

    part_finetune = args.get('part_finetune', False)
    if part_finetune:
        try:
            finetune_layers = args['finetune_layers']
            for name, param in net.named_parameters():
                param.requires_grad = False
                for layer in finetune_layers:
                    if layer in name:
                        param.requires_grad = True
        except Exception:
            raise Warning('part finetune failed. switch to full fine-tuning')

    logging.info('[Trainable: {:.2e}; Total: {:.2e}]'.format(
        sum(p.numel() for p in net.parameters() if p.requires_grad),
        sum(p.numel() for p in net.parameters()),
    ))
    logging.info(f"Num classes: {args['num_classes']}")

    optim = build_optimizer(net, args['optim'], **args['optim_conf'])
    lr_sch = build_scheduler(
        optim=optim,
        sch_name=args['scheduler'],
        **args['scheduler_conf'][args['scheduler']]
    )
    train_dl = stepDataloader(
        ds=train_ds,
        batch_size=args['batch_size'],
        total_step=args['max_step'],
        num_workers=args.get('num_workers', 4)
    )

    reinit = args.get('reinit', False)
    if reinit:
        reinit_conf = args.get('reinit_conf', {})
        acc_queue = FixedSizeQueue(reinit_conf.get('chunk_size', 20))
        acc_thres = reinit_conf.get('acc_thres', 95)

    # log_info = f"step: 0/{args['max_step']} [lr: {optim.param_groups[0]['lr']:.1e}]"
    # total, info = adall(net, test_ds, args, os.path.join(args['exp_dir'], 'step_0'))
    # overall_score = np.mean([s for s in total.values()])
    # logging.info(log_info + ' ' + ' '.join([f'[{k}: {total[k]:.2f} {info[k]}]' for k in info.keys()]))
    # for k in info.keys():
    #     for a, b in info[k].items():
    #         if isinstance(b, int) or isinstance(b, float):
    #             writer.add_scalar(f'{k}/{a}', b, global_step=0)
    # writer.add_scalar('overall', overall_score, global_step=0)
    # if overall_score > best_score:
    #     best_score = overall_score
    #     model_name = f"step_0_hmean_{best_score:0.3f}"
    #     torch.save(net.state_dict(), os.path.join(model_path, model_name))
    net.train()

    pbar = tqdm(train_dl, total=args['max_step'])
    for curstep, medata in pbar:
        '''
        x = medata['wav'].cuda()
        valid_len = medata['valid_len'].cuda()
        label = medata['classify_labs'].cuda()
        output_dict = net(
            x=x,
            valid_len=valid_len,
            label=label
        )
        '''
        bs, output_dict = net_forward(net, medata, args)
        '''
        loss = output_dict['loss']
        loss.backward()
        '''
        loss_item, acc_num = loss_backward(output_dict, args)
        
        if curstep % args['accumulate_grad'] == 0:
            optim.step()
            optim.zero_grad()
            if args['scheduler'] in [
                'WarmupLR',
                'WarmupLRLinear',
                'CosineAnnealingLR',
                'CosineAnnealingWarmupRestarts'
            ]:
                lr_sch.step()
        learning_curves.append(loss_item)
        #acc_num = output_dict['acc_num'].cpu().item()
        acc = acc_num / bs * 100
        desc = f"step: {curstep}/{args['max_step']} loss: {loss_item} lr: {optim.param_groups[0]['lr']:.1e} acc: {acc:.2f}"
        pbar.set_description(desc)
        writer.add_scalar('loss', loss_item, global_step=curstep)
        writer.add_scalar('lr', optim.param_groups[0]['lr'], global_step=curstep)
        writer.add_scalar('deputy_acc', acc, global_step=curstep)
        if curstep % 20 == 0:
            logging.info(desc)

        if reinit:
            acc_queue.put(acc)
            if acc_queue.get_aver() > acc_thres:
                with torch.no_grad():
                    net.loss.init_weight()
                logging.info('reinitialize predictor projection')

        if curstep % args['obInterval'] == 0:
            if args['scheduler'] in ['ReduceLROnPlateau']:
                avg_loss = np.mean(learning_curves[-args['obInterval']:])
                lr_sch.step(avg_loss)
            log_info = f"step: {curstep}/{args['max_step']} [lr: {optim.param_groups[0]['lr']:.1e}]"
            total, info = adall(net, test_ds, args, os.path.join(args['exp_dir'], f'step_{curstep}'))
            overall_score = np.mean([s for s in total.values()])
            logging.info(log_info + ' ' + ' '.join([f'[{k}: {total[k]:.2f} {info[k]}]' for k in info.keys()]))
            for k in info.keys():
                for a, b in info[k].items():
                    if isinstance(b, int) or isinstance(b, float):
                        writer.add_scalar(f'{k}/{a}', b, global_step=curstep)
            writer.add_scalar('overall', overall_score, global_step=curstep)
            
            if overall_score > best_score:
                best_score = overall_score
                best_it = curstep
                model_name = f"step_{curstep}_hmean_{best_score:0.3f}"
                torch.save(net.state_dict(), os.path.join(model_path, model_name))
                if use_lora:
                    if len(lora.lora_state_dict(net)) > 0:
                        torch.save(lora.lora_state_dict(net), os.path.join(lora_path, model_name))
                    else:
                        torch.save(lora.lora_state_dict(net.model), os.path.join(lora_path, model_name))
            # to save the iterations after 'best'
            elif (curstep-best_it) <= 500:
                model_name = f"step_{curstep}_hmean_{overall_score:0.3f}"
                torch.save(net.state_dict(), os.path.join(model_path, model_name))
                if use_lora:
                    if len(lora.lora_state_dict(net)) > 0:
                        torch.save(lora.lora_state_dict(net), os.path.join(lora_path, model_name))
                    else:
                        torch.save(lora.lora_state_dict(net.model), os.path.join(lora_path, model_name))
            net.train()
            
    pbar.close()


def adall(
    net: nn.Module,
    test_ds: Dict[str, Dataset],
    args: Dict[str, Any],
    exp_dir: str
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    total, info = {}, {}
    for k in test_ds.keys():
        testdl = DataLoader(
            test_ds[k],
            args.get('test_batch_size', args['batch_size'] * 4),
            shuffle=False,
            num_workers=args.get('num_workers', 4)
        )
        if 'dcase' in k:
            if args['mode'] == 'eval' and args['emb_dir'] is not None:
                emb_file = os.path.join(args['emb_dir'], k, 'emb.pkl')
                with open(emb_file, 'rb') as fp:
                    emb_data = pickle.load(fp)
            else:
                net.eval()
                seg_emb = {'file': [], 'emb': []}
                with torch.no_grad():
                    for medata in tqdm(testdl, desc='Extracting embeddings'):
                        '''
                        x = medata['wav'].cuda()
                        valid_len = medata['valid_len']
                        output_dict = net(
                            x=x,
                            valid_len=valid_len,
                            out_emb=True
                        )
                        '''
                        _, output_dict = net_forward(net, medata, args, True)
                        
                        seg_emb['emb'].append(output_dict['embedding'].cpu().numpy())
                        seg_emb['file'].extend(medata['file'])
                seg_emb['emb'] = np.concatenate(seg_emb['emb'], axis=0)

                emb_data = []
                file_list = sorted(list(set(seg_emb['file'])))
                seg_map = {f: [] for f in file_list}
                for i, f in enumerate(seg_emb['file']):
                    seg_map[f].append(i)
                for f in file_list:
                    file_emb = ensemble_embs(
                        embs=seg_emb['emb'][seg_map[f]],
                        ensem_mode=args.get('feat_aggre', args['aggregation'])
                    )
                    emb_data.append({'file': f, 'emb': file_emb})

                os.makedirs(os.path.join(exp_dir, k), exist_ok=True)
                with open(os.path.join(exp_dir, k, 'emb.pkl'), 'wb') as fp:
                    pickle.dump(emb_data, fp)

            DCASE_AD = DCASE_Anomaly_Detection(
                dataset=k,
                conf_dir=args['AD_conf_dir'],
                meta_data_dir=args['meta_data_dir'],
                score_aggre=args.get('score_aggre', None)
            )
            score_dict = DCASE_AD.score(emb_data)
            if args['mode'] == 'train':
                set_result, subset_total = cal_aucs(
                    score_dict=score_dict,
                    dataset=k,
                    exp_dir=exp_dir,
                    score_vis=False,
                    gen_sub=False
                )
            else:
                set_result, subset_total = cal_aucs(
                    score_dict=score_dict,
                    dataset=k,
                    exp_dir=exp_dir,
                    score_vis=False,
                    gen_sub=True
                )
            # select best detector for all machines
            set_total = {}
            for d in subset_total.keys():
                if 'eval' in subset_total[d].keys():
                    set_total[d] = hmean([subset_total[d]['dev'], subset_total[d]['eval']])
                    have_eval = True  # TODO:
                else:
                    set_total[d] = subset_total[d]['dev']
                    have_eval = False  # TODO:
            best = max(set_total.items(), key=lambda x: x[1])
            if have_eval:
                desc = {'dev': round(subset_total[best[0]]['dev'], 2),
                        'eval': round(subset_total[best[0]]['eval'], 2),
                        'detector': best[0]}
            else:
                desc = {'dev': round(subset_total[best[0]]['dev'], 2),
                        'detector': best[0]}
            total[k], info[k] = best[1], desc

            with open(os.path.join(exp_dir, k, 'score.pkl'), 'wb') as fp:
                pickle.dump(score_dict, fp)
            with open(os.path.join(exp_dir, k, 'result.pkl'), 'wb') as fp:
                pickle.dump(set_result[best[0]], fp)

        # elif k == 'idmt_engine':
        #     seg_result = {}  # {file：{label, pred: []}}
        #     for medata in tqdm(testdl, desc='validation'):
        #         with torch.no_grad():
        #             x = medata['wav'].cuda()
        #             output_dict = net(x)
        #             pred = output_dict['pred'].squeeze().cpu().numpy()
        #             for i in range(x.shape[0]):
        #                 file, label = medata['file'][i], medata['classify_labs'][i].item()
        #                 if file in seg_result:
        #                     seg_result[file]['pred'].append(pred[i])
        #                 else:
        #                     seg_result[file] = {'label': label, 'pred': [pred[i]]}

        #     acc_num, file_num = 0, len(seg_result)
        #     for file in seg_result:
        #         preds, counts = np.unique(np.asarray(seg_result[file]['pred']), return_counts=True)
        #         pred = preds[np.argmax(counts)]
        #         acc_num += seg_result[file]['label'] == pred
        #     acc = acc_num / file_num
        #     total[k], info[k] = acc, {}

    return total, info


def main(args: Dict[str, Any]) -> None:
    # exp dir
    os.makedirs(args['exp_dir'], exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args['exp_dir'], 'log'),
        format='%(asctime)s  %(levelname)s %(message)s',
        level=logging.INFO,
        force=True  # remove previous handlers
    )
    if args['mode'] == 'train':
        if args['downstream'] == 'dcase':
            for dataset in args['test_sets']:
                cmd = f"cp {os.path.join(args['AD_conf_dir'], dataset + '.yaml')} {args['exp_dir']}"
                os.system(cmd)
            args['AD_conf_dir'] = args['exp_dir']

    # data
    win_frames = int(args['input_duration'] * args['sample_rate'])
    hop_dur = args.get('hop_duration', None)
    if hop_dur is None:
        hop_size = win_frames
    else:
        hop_size = int(hop_dur * args['sample_rate'])
    if args['downstream'] == 'dcase':
        task_conf = args.get('task_conf', {})
        if args['task'] == 'machine':
            label_fn = DCASE_Machine_Labeler(
                dataset=args['train_sets'][0]
            )
        elif args['task'] == 'section':
            label_fn = DCASE_Section_Labeler()
        elif args['task'] == 'machine_section':
            label_fn = DCASE_Machine_Section_Labeler()
        elif args['task'] == 'attribute':
            label_fn = DCASE_Attribute_Labeler(**task_conf)
        elif args['task'] == 'pseudo':
            label_fn = DCASE_Pseudo_Labeler(**task_conf)
        else:
            raise NotImplementedError(f"Training task {args['task']} not implemented!")
    elif args['downstream'] == 'idmt_engine':
        label_fn = IDMT_Engine_Labeler
    else:
        raise NotImplementedError(f"Training task {args['task']} not implemented!")

    print('Loading data...')
    if args['downstream'] == 'dcase':
        train_ds, read_data = build_dcase_dataset(
            args=args,
            datasets=args['train_sets'],
            ds_mt=args['machine_type'],
            win_frames=win_frames,
            hop_size=None,
            label_fn=label_fn,
            cond=['train'],
            save_read_data=True
        )
        train_ds = train_ds[args['train_sets'][0]]
        args_test = copy.deepcopy(args)
        args_test['speed_perturb'] = False
        args_test['volume_perturb'] = False
        args_test['audio_roll'] = False
        args_test['spec_aug'] = False
        test_ds, _ = build_dcase_dataset(
            args=args_test,
            datasets=args['test_sets'],
            ds_mt={d: [] for d in args['test_sets']},
            win_frames=win_frames,
            hop_size=hop_size,
            label_fn=None,
            cond=[],
            pre_read_data=read_data
        )
    elif args['downstream'] == 'idmt_engine':
        test_ds = {}
        train_ds, test_ds['idmt_engine'] = build_idmt_engine_dataset(
            meta_data_dir=args['meta_data_dir'],
            win_frames=win_frames,
            hop_size=hop_size,
            label_fn=label_fn,
            sr=args['sample_rate']
        )

    # network
    print('Setting up network...')
    args['num_classes'] = train_ds.num_classes
    net = construct_model(args)
    net.cuda()

    if args['mode'] == 'train':
        logging.info(args)
        # tensorboard
        tb_top_dir = os.path.join(args['exp_dir'], 'tb')
        os.makedirs(tb_top_dir, exist_ok=True)
        prev_tb = [
            a for a in os.listdir(tb_top_dir)
            if os.path.isdir(os.path.join(tb_top_dir, a))
        ]
        writer = SummaryWriter(log_dir=os.path.join(tb_top_dir, str(len(prev_tb))))

        train_and_test(net, train_ds, test_ds, args, writer)
        writer.close()
    else:  # eval
        ckpt, load_weights, emb_dir = find_best_ckpt_emb(args)
        if emb_dir is not None:
            args['emb_dir'] = emb_dir
        elif (ckpt is not None) and load_weights:
            net_dict = torch.load(ckpt)
            missing_unexpected = net.load_state_dict(net_dict, strict=True)
            logging.info(missing_unexpected)
            use_lora = args.get('lora', False)
            if use_lora:
                lora_path = os.path.join(os.path.dirname(ckpt), 'lora', os.path.basename(ckpt))
                lora_dict = torch.load(lora_path)
                net.load_state_dict(lora_dict, strict=False)
        adall(net, test_ds, args, os.path.join(args['exp_dir'], args['result_dir']))

    if net is not None:
        net.cpu()
    del net
    del train_ds
    del test_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='conf/panns/24/cnn10.yaml')
    parser.add_argument('--basic_conf', type=str, default='conf/basic.yaml')
    parser.add_argument('--exp_dir', type=str, default='exp/cnn10/24/eval/attri/scratch/run0/')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--ckpt', type=str, help='relative to exp_dir', default=None)
    parser.add_argument('--emb_dir', type=str, help='relative to exp_dir', default=None)
    parser.add_argument('--result_dir', type=str, help='relative to exp_dir', default='none')
    parser.add_argument('--ntimes', type=int, help='number of runs', default=5)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    set_seed(args.seed)

    with open(args.conf, 'r', encoding='utf-8') as fp:
        conf = yaml.safe_load(fp)
    with open(args.basic_conf, 'r', encoding='utf-8') as fp:
        basic_conf = yaml.safe_load(fp)
    args = vars(args)
    args.update(conf)
    args.update(basic_conf)

    if args['mode'] == 'train':
        start_id = len(glob.glob(os.path.join(args['exp_dir'], 'run*')))
        for nrun in range(args['ntimes']):
            run_args = copy.deepcopy(args)
            run_args['exp_dir'] = os.path.join(run_args['exp_dir'], f'run{nrun + start_id}')
            main(run_args)
            gc.collect()  # release unused variables
    else:  # eval
        main(args)
