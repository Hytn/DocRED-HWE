import ujson as json
import pickle
from collections import defaultdict
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.misc import derivative
import pickle
from transformers import AutoTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="roberta-large", type=str)
map_args = parser.parse_args()
MAX_SEQ_LENGTH = 1024

rel2id_path = "dataset/meta/rel2id.json"
docred_rel2id = json.load(open(rel2id_path))
id2rel = {v: k for k, v in docred_rel2id.items()}
dev_keys_new = json.load(open("dataset/docred/dev_keys_new.json"))
kdict = pickle.load(open("dataset/docred/keywords_dict.pkl", "rb"))

model_type = map_args.model_type
tokenizer = AutoTokenizer.from_pretrained(model_type)

# read IG val list
ig_atl_all_list = pickle.load(open("dataset/ig_pkl/dev_keys_new_ig_all@roberta-large_atlop.pkl", "rb"))
# ig_docu_all_list = pickle.load(open("../DocuNet/infers/docu_dev_keys_new_ig_all.pkl", "rb"))


MAX_TOPK = 100
MIN_VAL = -1000000


def sharpen(arr, T):
    T = 1 / T
    sum_arr = np.sum(np.power(arr, T), axis=0)
    return arr / sum_arr


def cal_mAP(val_all_list, topk, limit=True, offset=0):
    data = dev_keys_new
    stop_i = len(val_all_list)  # for partial calculation
    no_num = 0
    mAPs = []
    old_mAPs = []
    sharps, max_topks = [], []
    if limit:
        data = data[:10]
        data = tqdm(data, desc="Keyword Search")
    for si, sample in enumerate(data):
        if si >= stop_i:
            break
        sents = []
        sent_map = []
        entities = sample["vertexSet"]
        entity_start, entity_end = [], []
        all_mentions = []

        for e_list in entities:
            all_mentions += e_list
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append(
                    (
                        sent_id,
                        pos[0],
                    )
                )
                entity_end.append(
                    (
                        sent_id,
                        pos[1] - 1,
                    )
                )
        for i_s, sent in enumerate(sample["sents"]):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece  # add * around entity
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        train_triple = {}
        if "labels" in sample:
            for label in sample["labels"]:
                evidence = label["evidence"] if "evidence" in label else []
                r = int(docred_rel2id[label["r"]])
                if (label["h"], label["t"]) not in train_triple:
                    train_triple[(label["h"], label["t"])] = [{"relation": r, "evidence": evidence}]
                else:
                    train_triple[(label["h"], label["t"])].append({"relation": r, "evidence": evidence})
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append(
                    (
                        start,
                        end,
                    )
                )
        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] + [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])

        key_dict = defaultdict(list)
        for h, t, r, sent_id, st, ed, name in kdict[sample["title"]]:
            key_dict[(h, t, r)].append((sent_id, st, ed, name))
        for h, t, r in key_dict.keys():
            ht_i = hts.index([h, t])
            ig_vals, topk_indices = val_all_list[si][0][ht_i], val_all_list[si][1][ht_i].tolist()

            ro = r + offset
            if ro not in topk_indices:  # skip wrong grad prediction
                continue

            # ind_topk = np.argpartition(ig_vals[topk_indices.index(r)], -topk, axis=-1)[-topk:].tolist()
            ind_topk = np.argsort(ig_vals[topk_indices.index(ro)], axis=-1)[-topk:].tolist()
            ind_topk.reverse()
            max_ind_topk = np.argpartition(ig_vals[topk_indices.index(ro)], -MAX_TOPK, axis=-1)[-MAX_TOPK:].tolist()
            max_ind_topk.reverse()
            # new MAP
            topk_np = ig_vals[topk_indices.index(ro), max_ind_topk]
            topk_sharp = np.flip(np.sort(sharpen(topk_np, 0.5)))
            sharps.append(topk_sharp)

            fx = np.linspace(0, MAX_TOPK, MAX_TOPK)
            f = interp1d(fx, topk_sharp, kind="cubic")
            dx = 1 / 10
            one = np.empty((MAX_TOPK))
            two = np.empty((MAX_TOPK))
            for fi in range(MAX_TOPK):
                one[fi] = derivative(f, 0 + dx + dx * fi, dx)  # second param is derivative position
                two[fi] = derivative(f, 0 + dx + dx * fi, dx, n=2)
            # find point x that one(x) ~= 0 and two(x) ~> 0
            mtopk = 0
            find_one = False
            for i in range(len(one) - 1):
                if one[i] * one[i + 1] < 0:
                    find_one = True
                    mtopk = i
                    break
            if not find_one:
                # to find 0 nearest value
                mtopk = np.argmin(np.abs(one))
                # print(f'not found one==0, sub: {mtopk}')
                no_num += 1
            for i in range(mtopk, len(two) - 1):
                if two[i] < 0 and two[i] * two[i + 1] < 0:
                    max_topks.append(i)
                    mtopk = i
                    break

            if len(ind_topk) != topk:
                raise ValueError
            key_inds = []
            for sent_id, st, ed, _ in key_dict[(h, t, r)]:
                st, ed = sent_map[sent_id][st], sent_map[sent_id][ed]
                key_inds.extend(range(st, ed))
            key_inds = set(key_inds)
            ap, old_ap = [], []
            for pos in range(len(ind_topk)):
                if ind_topk[pos] in key_inds:
                    pp = (pos + 1) if pos <= mtopk else (mtopk + 1)
                    ap.append(1 / pp)
                    old_ap.append(1 / (pos + 1))
            mAPs.append(sum(ap) / topk)
            old_mAPs.append(sum(old_ap) / topk)
    return sum(mAPs) / len(mAPs) if len(mAPs) > 0 else 0, sum(old_mAPs) / len(old_mAPs), len(mAPs), max_topks, no_num


ig_atl_nmap_stat, ig_docu_nmap_stat = [], []
for topk in trange(1, MAX_TOPK + 1):
    mAPs, old_mAPs = cal_mAP(ig_atl_all_list, topk, limit=False)[:2]
    ig_atl_nmap_stat.append(mAPs)

    # mAPs, old_mAPs = cal_mAP(ig_docu_all_list, topk, limit=False)[:2]
    # ig_docu_nmap_stat.append(mAPs)


pickle.dump(ig_atl_nmap_stat, open("dataset/keyword_pkl/ig_atl_nmap_stat.pkl", "wb"))
# pickle.dump(ig_docu_nmap_stat, open("dataset/keyword_pkl/ig_docu_nmap_stat.pkl", "wb"))
