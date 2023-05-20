import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from torch.nn import functional as F
import pickle
import ujson as json
from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model import DocREModel_infer
from utils import set_seed, collate_fn
from prepro import read_docred
from tqdm import trange, tqdm
from torch.autograd import grad
import time


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

ig_config = {
    'save_interval': 10,
    'step_batch_size': 2,
    'start_p': 0,
    'topk': 4,
    'appr_steps': 10  # 10
}
# seq_len --> batch_size
# 318: 10

def ig_infer_sp_bs(model, infer_features, args):
    print('use ig_infer_sp method')
    dataloader = DataLoader(infer_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    infer_name = args.infer_file.split('.')[0] + '@' + args.load_path.split('.')[0].split('/')[-1]
    sample_len = len(infer_features)
    debug_flag = '' if args.prod else '--debug'
    print("{}, sample num = {}".format(infer_name,sample_len))

    ig_pairs = []
    save_interval = ig_config['save_interval']
    steps = ig_config['appr_steps']
    k = ig_config['topk']  # top-k prob labels
    # start_i = ig_config['start_p']
    start_i = args.start_p
    bs = ig_config['step_batch_size']
    print('start p = ', start_i)
    # skip start samples
    si = 0

    # start_i=380  # debug

    # sample by sample
    for sample in tqdm(dataloader):
        if si >= 12: break
        if si < start_i:
            si += 1
            print(si)
            continue
        # input_ids, attention_mask, labels, bert_entity_pos, hts, entity2mention, mention_num, vert_type, title
        stime = time.time() 
        input_ids = sample[0].to(args.device)

        # baseline generation (use 0 embedding value)
        input_embs = model.model.get_input_embeddings()(input_ids)
        inputs = {'input_embs': input_embs.to(args.device),
                  'attention_mask': sample[1].to(args.device),
                  'labels': sample[2],
                  'entity_pos': sample[3],
                  'hts': sample[4],
                  }
        logits = model(**inputs)

        baseline_input_embs = torch.zeros_like(input_embs)
        topk_logits, topk_indices = torch.topk(logits[:,1:], k, dim=-1)  # remove threshold class
        scale_input_embs = [baseline_input_embs + (step / steps) * (
                input_embs - baseline_input_embs) for step in range(1, steps + 1)]
        # to batch
        scale_inp_batch = []
        # TODO: adaptive batch_size
        seq_len = input_embs.shape[1]
        # print('sample: %d, seq_len: %d' % (si, seq_len))
        if seq_len < 400: # 318
            bs = 10
        elif seq_len <= 500: # 450 cor
            bs = 5
        else: # > 638
            bs = 4
        for i in range(0,len(scale_input_embs), bs):
            scale_inp = torch.cat(scale_input_embs[i:i+bs])
            size = scale_inp.shape[0]
            attention_mask = sample[1].expand(size, -1)
            entity_pos = sample[3] * size
            hts = sample[4] * size
            scale_inp_batch.append((scale_inp, attention_mask, entity_pos, hts, size))

        delta_input = input_embs - baseline_input_embs
        delta_input = delta_input.squeeze(0).cpu()

        rel_num = topk_logits.shape[0]  # relation pair num

        # shape = rel_num * k * scale_num * seq_len * hidden_size
        rel_ig = torch.zeros(rel_num, k, delta_input.shape[0], delta_input.shape[1])
        for scale_inp, attention_mask, entity_pos, hts, size in scale_inp_batch:
            # batch of inp_embs
            inputs = {'input_embs': scale_inp.to(args.device),
                        'attention_mask': attention_mask.to(args.device),
                        # 'labels': sample[2],
                        'entity_pos': entity_pos,
                        'hts': hts,
                        }
            logits= model(**inputs) # [concat rel_num, rel_i]  
            for rel_i in range(0, rel_num):
                for k_i in range(k):  # topk           
                    # model.zero_grad()
                    # label_logit = logits[rel_i, topk_indices[rel_i, k_i]]
                    # parallel computing of grad()
                    grad_output = torch.zeros_like(logits)
                    grad_output[[ind * rel_num for ind in range(size)], topk_indices[rel_i, k_i]] = 1
                    gradient = grad(outputs=logits, inputs=scale_inp, retain_graph=True,grad_outputs=grad_output)[0].squeeze(0)
                    rel_ig[rel_i, k_i] += gradient.sum(0).detach_().cpu() # detach_() change tensor itself
            # release CUDA memory
            # del gradient
            # del scale_inp
            del logits
            del grad_output
            torch.cuda.empty_cache()
        with torch.no_grad():
            rel_ig = rel_ig * delta_input / steps
            rel_ig = rel_ig.sum(-1)
            rel_ig = rel_ig.numpy()
        print('end one sample time = ', time.time() - stime, ' sec')
        topk_indices = topk_indices.detach().cpu().numpy()  # skip threshold class
        ig_pairs.append((rel_ig,topk_indices))  # store in pair list
        file_path = os.path.join(args.ig_dir, infer_name +'_ig_' + str(start_i) + 'to' + str(si) + debug_flag + '.pkl')
        si += 1
        with open(file_path, 'wb') as f:
            pickle.dump(ig_pairs, f)

    # # store to disk
    # with open(os.path.join(args.ig_dir, infer_name + '_ig_all' + debug_flag +'.pkl'), 'wb') as f:
    #     pickle.dump(ig_pairs, f)
    
def ig_infer_sp_bs_pad(model, infer_features, args):
    print('use ig_infer_sp method')
    dataloader = DataLoader(infer_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    infer_name = args.infer_file.split('.')[0] + '@' + args.load_path.split('.')[0].split('/')[-1]
    sample_len = len(infer_features)
    debug_flag = '' if args.prod else '--debug'
    print("{}, sample num = {}".format(infer_name,sample_len))

    ig_pairs = []
    save_interval = ig_config['save_interval']
    steps = ig_config['appr_steps']
    k = ig_config['topk']  # top-k prob labels
    start_i = ig_config['start_p']
    bs = ig_config['step_batch_size']
    print('start p = ', start_i)
    # skip start samples
    si = 0

    word_embedding = model.model.get_input_embeddings()
    model = torch.nn.DataParallel(model, device_ids=[0,1]) # error 

    # sample by sample
    for sample in tqdm(dataloader):
        if si < start_i:
            si += 1
            print(si)
            continue
        # input_ids, attention_mask, labels, bert_entity_pos, hts, entity2mention, mention_num, vert_type, title
        stime = time.time() 
        input_ids = sample[0].to(args.device)

        pad_num = 80
        input_ids = F.pad(input_ids, (0,pad_num),value=103)

        # baseline generation (use 0 embedding value)
        input_embs = word_embedding(input_ids)
        inputs = {'input_embs': input_embs.to(args.device),
                #   'attention_mask': sample[1].to(args.device),
                  'attention_mask': F.pad(sample[1],(0,pad_num),value=1).to(args.device),
                  'labels': sample[2],
                  'entity_pos': sample[3],
                  'hts': sample[4],
                  }
        logits = model(**inputs)

        baseline_input_embs = torch.zeros_like(input_embs)
        topk_logits, topk_indices = torch.topk(logits[:,1:], k, dim=-1)  # remove threshold class
        scale_input_embs = [baseline_input_embs + (step / steps) * (
                input_embs - baseline_input_embs) for step in range(1, steps + 1)]
        # to batch
        scale_inp_batch = []
        # TODO: adaptive batch_size
        seq_len = input_embs.shape[1]
        # print('sample: %d, seq_len: %d' % (si, seq_len))
        if seq_len <= 400: # 318
            bs = 10
        elif seq_len <= 500: # 450 cor
            bs = 5
        else: # > 638
            bs = 4
        for i in range(0,len(scale_input_embs), bs):
            scale_inp = torch.cat(scale_input_embs[i:i+bs])
            size = scale_inp.shape[0]
            attention_mask = sample[1].expand(size, -1)
            attention_mask = F.pad(attention_mask,(0,pad_num),value=1)
            entity_pos = sample[3] * size
            hts = sample[4] * size
            scale_inp_batch.append((scale_inp, attention_mask, entity_pos, hts, size))

        delta_input = input_embs - baseline_input_embs
        delta_input = delta_input.squeeze(0).cpu()

        rel_num = topk_logits.shape[0]  # relation pair num

        # shape = rel_num * k * scale_num * seq_len * hidden_size
        rel_ig = torch.zeros(rel_num, k, delta_input.shape[0], delta_input.shape[1])
        for scale_inp, attention_mask, entity_pos, hts, size in scale_inp_batch:
            # batch of inp_embs
            inputs = {'input_embs': scale_inp.to(args.device),
                        'attention_mask': attention_mask.to(args.device),
                        # 'labels': sample[2],
                        'entity_pos': entity_pos,
                        'hts': hts,
                        }
            logits= model(**inputs) # [bs, rel_num, rel_i]
            for rel_i in range(0, rel_num):
                for k_i in range(k):  # topk           
                    # model.zero_grad()
                    # label_logit = logits[rel_i, topk_indices[rel_i, k_i]]
                    # parallel computing of grad()
                    grad_output = torch.zeros_like(logits)
                    grad_output[[ind * rel_num for ind in range(size)], topk_indices[rel_i, k_i]] = 1
                    gradient = grad(outputs=logits, inputs=scale_inp, retain_graph=True,grad_outputs=grad_output)[0].squeeze(0)
                    rel_ig[rel_i, k_i] += gradient.sum(0).detach_().cpu() # detach_() change tensor itself
            # release CUDA memory
            # del gradient
            # del scale_inp
            del logits
            del grad_output
            torch.cuda.empty_cache()
        with torch.no_grad():
            rel_ig = rel_ig * delta_input / steps
            rel_ig = rel_ig.sum(-1)
            rel_ig = rel_ig.numpy()
        print('end one sample time = ', time.time() - stime, ' sec')
        topk_indices = topk_indices.detach().cpu().numpy()  # skip threshold class
        ig_pairs.append((rel_ig,topk_indices))  # store in pair list
        file_path = os.path.join(args.ig_dir, infer_name +'_ig_' + str(si) + debug_flag + '.pkl')
        si += 1
        with open(file_path, 'wb') as f:
            pickle.dump(ig_pairs, f)

    # store to disk
    with open(os.path.join(args.ig_dir, infer_name + '_ig_all' + debug_flag +'.pkl'), 'wb') as f:
        pickle.dump(ig_pairs, f)

def ig_infer_sp(model, infer_features, args):
    print('use ig_infer_sp method')
    dataloader = DataLoader(infer_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    infer_name = args.infer_file.split('.')[0] + '@' + args.load_path.split('.')[0].split('/')[-1]
    sample_len = len(infer_features)
    print("{}, sample num = {}".format(infer_name,sample_len))

    ig_pairs = []
    save_interval = ig_config['save_interval']
    steps = ig_config['appr_steps']
    k = ig_config['topk']  # top-k prob labels
    start_i = ig_config['start_p']
    print('start p = ', start_i)
    # skip start samples
    si = 0

    # sample by sample
    for sample in tqdm(dataloader):
        if si < start_i:
            si += 1
            print(si)
            continue
        # input_ids, attention_mask, labels, bert_entity_pos, hts, entity2mention, mention_num, vert_type, title
        stime = time.time() 
        input_ids = sample[0].to(args.device)

        # baseline generation (use 0 embedding value)
        input_embs = model.model.get_input_embeddings()(input_ids)
        inputs = {'input_embs': input_embs.to(args.device),
                  'attention_mask': sample[1].to(args.device),
                  'labels': sample[2],
                  'entity_pos': sample[3],
                  'hts': sample[4],
                  }
        logits = model(**inputs)

        baseline_input_embs = torch.zeros_like(input_embs)
        topk_logits, topk_indices = torch.topk(logits[:, 1:], k, dim=-1)  # remove threshold class
        scale_input_embs = [baseline_input_embs + (step / steps) * (
                input_embs - baseline_input_embs) for step in range(1, steps + 1)]
        delta_input = input_embs - baseline_input_embs
        delta_input = delta_input.squeeze(0)

        rel_num = topk_logits.shape[0]  # relation pair num

        # shape = rel_num * k * scale_num * seq_len * hidden_size
        rel_ig = torch.zeros(rel_num, k, delta_input.shape[0], delta_input.shape[1]).to(args.device)
        for inp_embs in scale_input_embs:
            inputs = {'input_embs': inp_embs.to(args.device),
                        'attention_mask': sample[1].to(args.device),
                        'labels': sample[2],
                        'entity_pos': sample[3],
                        'hts': sample[4],
                        }
            logits= model(**inputs)
            for rel_i in range(rel_num):
                for k_i in range(k):  # topk           
                    # model.zero_grad()
                    label_logit = logits[rel_i, topk_indices[rel_i, k_i]]
                    # parallel computing of grad()
                    gradient = grad(outputs=label_logit, inputs=inp_embs, retain_graph=True)[0].squeeze(0)
                    rel_ig[rel_i, k_i] += gradient
        rel_ig = rel_ig * delta_input / steps
        rel_ig = rel_ig.sum(-1)
        rel_ig = rel_ig.detach().cpu().numpy()
        print('end one sample time = ', time.time() - stime, ' sec')
        topk_indices = topk_indices.detach().cpu().numpy()  # skip threshold class
        ig_pairs.append((rel_ig,topk_indices))  # store in pair list
        file_path = os.path.join(args.ig_dir, infer_name +'_ig_' + str(si) + '.pkl')
        si += 1
        with open(file_path, 'wb') as f:
            pickle.dump(ig_pairs, f)

    # store to disk
    with open(os.path.join(args.ig_dir, infer_name + '_ig_all.pkl'), 'wb') as f:
        pickle.dump(ig_pairs, f)


def ig_infer(model, infer_features, args):
    print('use ig_infer method')
    dataloader = DataLoader(infer_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    infer_name = args.infer_file.split('.')[0] + '@' + args.load_path.split('.')[0].split('/')[-1]
    sample_len = len(infer_features)
    print("{}, sample num = {}".format(infer_name,sample_len))

    ig_pairs = []
    save_interval = ig_config['save_interval']
    steps = ig_config['appr_steps']
    k = ig_config['topk']  # top-k prob labels
    start_i = ig_config['start_p']
    print('start p = ', start_i)
    # skip start samples
    si = 0

    # sample by sample
    for sample in tqdm(dataloader):
        if si < start_i:
            si += 1
            print(si)
            continue
        stime = time.time()
        # input_ids, attention_mask, labels, bert_entity_pos, hts, entity2mention, mention_num, vert_type, title
        input_ids = sample[0].to(args.device)

        # baseline generation (use 0 embedding value)
        input_embs = model.model.get_input_embeddings()(input_ids)
        inputs = {'input_embs': input_embs.to(args.device),
                  'attention_mask': sample[1].to(args.device),
                  'labels': sample[2],
                  'entity_pos': sample[3],
                  'hts': sample[4],
                  }
        logits = model(**inputs)

        baseline_input_embs = torch.zeros_like(input_embs)
        topk_logits, topk_indices = torch.topk(logits[:,1:], k, dim=-1)  # remove threshold class
        scale_input_embs = [baseline_input_embs + (step / steps) * (
                input_embs - baseline_input_embs) for step in range(1, steps + 1)]
        delta_input = input_embs - baseline_input_embs

        rel_igs = []
        rel_num = topk_logits.shape[0]  # relation pair num
        # tr = trange(rel_num)
        # tr.set_postfix({'samples': '{}/{}'.format(sample_i, len(infer_features) - 1)})
        for rel_i in range(rel_num):
            k_igs = []
            for k_i in range(k):  # topk
                grads = []
                for inp_embs in scale_input_embs:
                    inputs = {'input_embs': inp_embs.to(args.device),
                              'attention_mask': sample[1].to(args.device),
                              'labels': sample[2],
                              'entity_pos': sample[3],
                              'hts': sample[4],
                              }
                    logits= model(**inputs)
                    # inp_embs.register_hook(set_grad(inp_embs))
                    model.zero_grad()

                    label_logit = logits[rel_i, topk_indices[rel_i, k_i]]
                    gradient = grad(outputs=label_logit, inputs=inp_embs, allow_unused=True,retain_graph=True)[0]
                    grads.append(gradient)
                    # use cat cover squeeze operation
                # sum all scale_inp's grad
                avg_grads = torch.cat(grads, dim=0).mean(dim=0).squeeze(0)
                # approximation of integral
                integrated_grad = delta_input.squeeze(0) * avg_grads
                k_igs.append(integrated_grad.sum(-1))  # sum all h_dim
            k_igs = torch.stack(k_igs)
            rel_igs.append(k_igs)
        rel_igs = torch.stack(rel_igs)  # rel_num * k * scale_num * seq_len * hidden_size

        print('end one sample time = ', time.time() - stime, ' sec')

        rel_igs = rel_igs.detach().cpu().numpy()
        topk_indices = topk_indices.detach().cpu().numpy()  # skip threshold class
        ig_pairs.append((rel_igs,topk_indices))  # store in pair list
        file_path = os.path.join(args.ig_dir, infer_name +'_ig_' + str(si) + '.pkl')
        si += 1
        with open(file_path, 'wb') as f:
            pickle.dump(ig_pairs, f)

    # store to disk
    with open(os.path.join(args.ig_dir, infer_name + '_ig_all.pkl'), 'wb') as f:
        pickle.dump(ig_pairs, f)



def grad_infer(model, infer_features, args):
    print('use grad_infer method')
    dataloader = DataLoader(infer_features, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
    infer_name = args.infer_file.split('.')[0]
    sample_len = len(infer_features)
    print("{}, sample num = {}".format(infer_name,sample_len))

    grad_pairs = []
    k = ig_config['topk']  # top-k prob labels
    start_i = ig_config['start_p']
    # skip start samples
    si = 0

    # sample by sample
    for sample in tqdm(dataloader):
        if si < start_i:
            si += 1
            continue
        # input_ids, attention_mask, labels, bert_entity_pos, hts, entity2mention, mention_num, vert_type, title
        input_ids = sample[0].to(args.device)

        # baseline generation (use 0 embedding value)
        input_embs = model.model.get_input_embeddings()(input_ids)
        inputs = {'input_embs': input_embs.to(args.device),
                  'attention_mask': sample[1].to(args.device),
                  'labels': sample[2],
                  'entity_pos': sample[3],
                  'hts': sample[4],
                  }
        logits = model(**inputs)
        topk_logits, topk_indices = torch.topk(logits[:,1:], k, dim=-1)  # remove threshold class

        rel_grads = []
        rel_num = topk_logits.shape[0]  # relation pair num
        for rel_i in range(rel_num):
            k_grads = []
            for k_i in range(k):  # topk
                grads = []
                model.zero_grad()

                label_logit = logits[rel_i, topk_indices[rel_i, k_i]]
                gradient = grad(outputs=label_logit, inputs=input_embs, allow_unused=True,retain_graph=True)[0]
                grads.append(gradient)
                # use cat cover squeeze operation
                # sum all scale_inp's grad
                avg_grads = torch.cat(grads, dim=0).mean(dim=0).squeeze(0)
                # approximation of integral
                k_grads.append(avg_grads.sum(-1))  # sum all h_dim
            k_grads = torch.stack(k_grads)
            rel_grads.append(k_grads)
        rel_grads = torch.stack(rel_grads)  # rel_num * k * scale_num * seq_len * hidden_size

        rel_grads = rel_grads.detach().cpu().numpy()
        topk_indices = topk_indices.detach().cpu().numpy()  # skip threshold class
        grad_pairs.append((rel_grads,topk_indices))  # store in pair list
        file_path = os.path.join(args.ig_dir, infer_name +'_reemb_grad_' + str(start_i)+'to'+ str(si) + '.pkl')
        si += 1
        with open(file_path, 'wb') as f:
            pickle.dump(grad_pairs, f)

    # store to disk
    with open(os.path.join(args.ig_dir, infer_name + '_reemb_grad_all.pkl'), 'wb') as f:
        pickle.dump(grad_pairs, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--ig_dir", default="grad_pkl", type=str)
    parser.add_argument("--start_p", default=0, type=int)
    parser.add_argument("--prod", action="store_true")
    parser.add_argument("--infer_file", default="dev_keys_new.json", type=str)
    
    parser.add_argument("--infer_method", default="grad_infer", type=str)
    parser.add_argument("--save_path", default="", type=str)
    # parser.add_argument("--load_path", default="saved_dict/model_ro_reemb_nofreeze.ckpt", type=str)
    parser.add_argument("--load_path", default="saved_dict/model_ro_reemb_nofreeze.ckpt", type=str)


    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--num_labels", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    BERT_PATH_DIR = '/cpfs/user/cht/cbsp/'
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else BERT_PATH_DIR+args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else BERT_PATH_DIR+args.model_name_or_path,
    )

    read = read_docred
    limit = False
    if args.prod: limit = False
    print(args)
    infer_file = os.path.join(args.data_dir, args.infer_file)
    infer_features = read(infer_file, tokenizer, max_seq_length=args.max_seq_length, limit=limit) \
        if len(args.infer_file) > 0 else None

    model = AutoModel.from_pretrained(
        BERT_PATH_DIR+args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel_infer(config, model, num_labels=args.num_labels)
    model.to(0)
    model = amp.initialize(model, opt_level="O1", verbosity=0)
    model.load_state_dict(torch.load(args.load_path, map_location=device))
    print("load saved model from {}.".format(args.load_path))

    infer_methods = {
        'ig_infer': ig_infer,
        'ig_infer_sp': ig_infer_sp,
        'ig_infer_sp_bs': ig_infer_sp_bs,
        'grad_infer':grad_infer,
    }
    print('using infer method: ', args.infer_method)
    infer_methods[args.infer_method](model, infer_features, args)
    # ig_infer_sp_bs_pad(model, infer_features, args)


if __name__ == "__main__":
    main()
