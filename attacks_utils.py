from tqdm import tqdm, trange
import ujson as json
import numpy as np
from collections import defaultdict
from itertools import groupby
import pickle
from nltk.corpus import wordnet as wn
from evaluation import evaluate, report
from prepro import read_docred


MAX_SEQ_LENGTH = 1024
MASK_VAL = -100000
rel2id_path = "dataset/meta/rel2id.json"
docred_rel2id = rel2id = json.load(open(rel2id_path))
id2rel = {v: k for k, v in rel2id.items()}


def enp_topk(args, model, model_type, tokenizer, ig_pkl_path, file_in="dataset/docred/dev_keys_new.json", limit=False):
    with open(ig_pkl_path, "rb") as dig:
        dev_keys_ig = pickle.load(dig)
    print(len(dev_keys_ig))
    # for enp_topk in trange(1,101):
    right = 1 if limit else 101

    print(f"start generating dataset from {file_in}")
    for enp_topk in trange(0, right):
        pos_samples = 0
        enp_features = []

        enp_num = 0
        with open(file_in, "r") as fh:
            data = json.load(fh)

        if limit:
            data = data[:10]
        # enp_topk = 100
        dev_key_enp_json = []
        #     for si, sample in enumerate(tqdm(data, desc="Example")):
        for si, sample in enumerate(data):
            sents = []
            sent_map = []
            entities = sample["vertexSet"]
            entity_start, entity_end = [], []
            all_mentions = []

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
                        train_triple[(label["h"], label["t"])] = [
                            {"relation": r, "rel": label["r"], "evidence": evidence}
                        ]
                    else:
                        train_triple[(label["h"], label["t"])].append(
                            {"relation": r, "rel": label["r"], "evidence": evidence}
                        )
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
                pos_samples += 1

            # all entity mention and position
            for ei, e in enumerate(entity_pos):
                for mi, m in enumerate(e):
                    all_mentions.append((m[0], m[1], ei, mi))  # e_ind and m_ind
            all_mentions.sort(key=lambda x: x[0])

            # 2) entity pair and topk words union for context
            hti = 0
            for h, t in train_triple.keys():
                ht_igs = dev_keys_ig[si][0][hti]
                assert ht_igs.shape[-1] == len(sents) + 2
                ht_sents, en_inds = [], []
                ht_entity_pos = [[], []]
                for m in entity_pos[h]:
                    en_inds.extend(list(range(m[0], m[1])))  # IG topk avoid en
                    ht_entity_pos[0].append((len(ht_sents), len(ht_sents) + m[1] - m[0]))
                    ht_sents.extend(sents[m[0] : m[1]])
                for m in entity_pos[t]:
                    en_inds.extend(list(range(m[0], m[1])))
                if enp_topk > 0:
                    # offset of model
                    en_inds = [ei + 1 for ei in list(set(en_inds))] + [
                        0,
                        ht_igs.shape[-1] - 1,
                    ]  # remove [CLS] and [SEP]
                    # IG topk context tokens (1.sum of all rel IG. vs  2.fetch union of 4 rels)
                    ht_igs[:, en_inds] = MASK_VAL
                    aind = np.argpartition(ht_igs.sum(0), -enp_topk, axis=-1)[-enp_topk:]
                    aind = aind - 1
                    aind = aind.tolist()
                    aind.sort()
                    ht_sents.extend([sents[idx] for idx in aind])
                for m in entity_pos[t]:
                    en_inds.extend(list(range(m[0], m[1])))  # IG topk avoid en
                    ht_entity_pos[1].append((len(ht_sents), len(ht_sents) + m[1] - m[0]))
                    ht_sents.extend(sents[m[0] : m[1]])

                ht_sents = ht_sents[: MAX_SEQ_LENGTH - 2]
                ht_input_ids = tokenizer.convert_tokens_to_ids(ht_sents)
                ht_input_ids = tokenizer.build_inputs_with_special_tokens(ht_input_ids)
                ht_title = f'{sample["title"]}_{h}_{t}'
                ht_feature = {
                    "input_ids": ht_input_ids,
                    "entity_pos": ht_entity_pos,
                    "hts": [[0, 1]],
                    "labels": [relations[hti]],
                    "title": ht_title,
                }
                enp_features.append(ht_feature)
                #         print(ht_feature,'\n')
                labels = []
                for l in train_triple[(h, t)]:
                    label = {"h": 0, "t": 1, "r": l["rel"], "evidence": l["evidence"]}
                    labels.append(label)
                dev_key_enp_json.append({"vertexSet": [entities[h], entities[t]], "labels": labels, "title": ht_title})
                hti += 1
                enp_num += 1

        pickle.dump(enp_features, open(f"dataset/docred/enp_topk/enp_topk{enp_topk}@{model_type}.pkl", "wb"))
        json.dump(dev_key_enp_json, open(f"dataset/docred/enp_topk/dev_key_enp_topk{enp_topk}.json", "w"))

    # start evaluation for entity pair and topk tokens
    enp_stat = []
    print("start evaluation...")

    for enp_topk in trange(0, 101):
        args.dev_file = f"enp_topk/dev_key_enp_topk{enp_topk}.json"
        enp_features = pickle.load(open(f"dataset/docred/enp_topk/enp_topk{enp_topk}.pkl", "rb"))
        bf, outp = evaluate(args, model, enp_features, tag=f"dev_enp_topk{enp_topk}")
        enp_stat.append(list(outp.values())[0])
    print(len(enp_stat))
    print(enp_stat)

    # save stat
    with open(f"enp_topk_stat@{model_type}.pkl", "rb") as dig:
        pickle.dump(enp_stat, dig)
    # draw graph
    print("start drawing figure for entity pair and topk tokens...")
    import matplotlib.pyplot as plt

    xaixs = range(0, 101, 10)
    fs = 30
    label_font = {"size": 35}
    plt.figure(figsize=(15, 10))
    plt.xlabel("topk", label_font)
    plt.ylabel("F1", label_font)
    plt.xticks(xaixs, fontsize=fs, rotation=20)
    plt.yticks(fontsize=fs)
    plt.plot(range(len(enp_stat)), enp_stat, label="ATLOP_roberta", color="darkgoldenrod")
    plt.legend(fontsize=35)
    plt.savefig(f"enp_topk@{model_type}.png")


def build_entity_attack_dataset(model_type, tokenizer, limit=False):
    # entities outside of DocRED
    with open("repl_ens.pkl", "rb") as e:
        repl_ens = pickle.load(e)
    print("repl_ens len = ", len(repl_ens))
    # 1. create [MASK] for all entity
    # 2. entity shuffle with next entity( all mention to the first mention)
    # 3. entity replaced with entity that out of distribution (NYT temporarily)
    file_in = "dataset/docred/dev.json"
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    ori_features, en_mask_features, en_shuf_features, en_repl_features = [], [], [], []
    MAX_SEQ_LENGTH = 1024

    MASK_TOKEN = tokenizer.mask_token
    ori_num = 0
    with open(file_in, "r") as fh:
        data = json.load(fh)
    if limit:
        data = data[:1]
    entity_type = ["-", "ORG", "-", "LOC", "-", "TIME", "-", "PER", "-", "MISC", "-", "NUM"]  # specific for DocuNet
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        entities = sample["vertexSet"]
        entity_start, entity_end = [], []
        mention_types = []
        all_mentions = []

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
                mention_types.append(mention["type"])

        for i_s, sent in enumerate(sample["sents"]):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    t = entity_start.index((i_s, i_t))
                    mention_type = mention_types[t]
                    special_token_i = entity_type.index(mention_type)
                    special_token = ["[unused" + str(special_token_i) + "]"]
                    tokens_wordpiece = special_token + tokens_wordpiece
                    # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    t = entity_end.index((i_s, i_t))
                    mention_type = mention_types[t]
                    special_token_i = entity_type.index(mention_type) + 50
                    special_token = ["[unused" + str(special_token_i) + "]"]
                    tokens_wordpiece = tokens_wordpiece + special_token
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
            relation = [0] + [0] * (len(docred_rel2id) - 1)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    # relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        # all entity mention and position
        for ei, e in enumerate(entity_pos):
            for mi, m in enumerate(e):
                all_mentions.append((m[0], m[1], ei, mi))  # e_ind and m_ind
        all_mentions.sort(key=lambda x: x[0])

        # ori feature
        ori_sents = sents.copy()
        ori_sents = ori_sents[: MAX_SEQ_LENGTH - 2]
        ori_input_ids = tokenizer.convert_tokens_to_ids(ori_sents)
        ori_input_ids = tokenizer.build_inputs_with_special_tokens(ori_input_ids)
        ori_feature = {
            "input_ids": ori_input_ids,
            "entity_pos": entity_pos,
            "hts": hts,
            "labels": relations,
            "title": sample["title"],
        }
        ori_num += 1
        ori_features.append(ori_feature)

        # 1)entity mask
        en_mask_sents = sents.copy()
        for e in entity_pos:
            for m in e:
                for i in range(m[0] + 1, m[1] - 1):  # skip [unused] token
                    en_mask_sents[i] = MASK_TOKEN
        en_mask_sents = en_mask_sents[: MAX_SEQ_LENGTH - 2]
        en_mask_input_ids = tokenizer.convert_tokens_to_ids(en_mask_sents)
        en_mask_input_ids = tokenizer.build_inputs_with_special_tokens(en_mask_input_ids)
        en_mask_feature = {
            "input_ids": en_mask_input_ids,
            "entity_pos": entity_pos,
            "hts": hts,
            "labels": relations,
            "title": sample["title"],
        }
        en_mask_features.append(en_mask_feature)

        # 2)entity shuffle (move to next entity position)
        en_shuf_sents = []
        shuf_ens = []
        shuf_index_now = 0
        for e in entity_pos:
            shuf_ens.append(sents[e[0][0] : e[0][1]])
        shuf_ens = shuf_ens[-1:] + shuf_ens[:-1]  # rotate to shuffle
        shuf_en_idx = 0
        shuf_entity_pos = []
        shuf_mentions = []
        for m in all_mentions:
            en_shuf_sents.extend(sents[shuf_index_now : m[0]])
            new_en = shuf_ens[m[2]]
            en_shuf_sents.extend(new_en)
            shuf_index_now = m[1]
            shuf_mentions.append((len(en_shuf_sents), len(en_shuf_sents) + len(new_en), m[2], m[3]))
        en_shuf_sents.extend(sents[shuf_index_now:])
        shuf_mentions.sort(key=lambda x: x[2])
        shuf_mens = [
            sorted([(s[0], s[1], s[-1]) for s in g], key=lambda y: y[-1])
            for k, g in groupby(shuf_mentions, key=lambda x: x[2])
        ]
        shuf_entity_pos = [[(s[0], s[1]) for s in ss] for ss in shuf_mens]

        en_shuf_sents = en_shuf_sents[: MAX_SEQ_LENGTH - 2]
        en_shuf_input_ids = tokenizer.convert_tokens_to_ids(en_shuf_sents)
        en_shuf_input_ids = tokenizer.build_inputs_with_special_tokens(en_shuf_input_ids)
        en_shuf_feature = {
            "input_ids": en_shuf_input_ids,
            "entity_pos": shuf_entity_pos,
            "hts": hts,
            "labels": relations,
            "title": sample["title"],
        }
        en_shuf_features.append(en_shuf_feature)

        # 3)entity replacement (using ood entity name)
        en_repl_sents = []
        repl_index_now = 0
        # TODO: shuffle or with fixed order ?
        repl_mentions = []
        for m in all_mentions:
            en_repl_sents.extend(sents[repl_index_now : m[0]])
            new_en = repl_ens[m[2] * 10]  # fetch every 10 entities
            en_wordpiece = tokenizer.tokenize(new_en)
            new_en = (
                [tokenizer.unk_token] + en_wordpiece + [tokenizer.unk_token]
            )  # <unk> is only for roberta (special token)
            en_repl_sents.extend(new_en)
            repl_index_now = m[1]
            repl_mentions.append((len(en_repl_sents), len(en_repl_sents) + len(new_en), m[2], m[3]))
        en_repl_sents.extend(sents[repl_index_now:])
        repl_mentions.sort(key=lambda x: x[2])
        repl_mens = [
            sorted([(s[0], s[1], s[-1]) for s in g], key=lambda y: y[-1])
            for k, g in groupby(repl_mentions, key=lambda x: x[2])
        ]
        repl_entity_pos = [[(s[0], s[1]) for s in ss] for ss in repl_mens]

        en_repl_sents = en_repl_sents[: MAX_SEQ_LENGTH - 2]
        en_repl_input_ids = tokenizer.convert_tokens_to_ids(en_repl_sents)
        en_repl_input_ids = tokenizer.build_inputs_with_special_tokens(en_repl_input_ids)
        en_repl_feature = {
            "input_ids": en_repl_input_ids,
            "entity_pos": repl_entity_pos,
            "hts": hts,
            "labels": relations,
            "title": sample["title"],
        }
        en_repl_features.append(en_repl_feature)
        i_line += 1

    attack_dir = "attack_pkl/"
    pickle.dump(ori_features, open(attack_dir + model_type + "@ori_dev.pkl", "wb"))
    pickle.dump(en_mask_features, open(attack_dir + model_type + "@en_mask_dev.pkl", "wb"))
    pickle.dump(en_shuf_features, open(attack_dir + model_type + "@en_shuf_dev.pkl", "wb"))
    pickle.dump(en_repl_features, open(attack_dir + model_type + "@en_repl_dev.pkl", "wb"))
    print("# of documents {}.".format(i_line))
    print("# of original num {}.".format(ori_num))
    print("# of shuffle num {}.".format(len(en_shuf_features)))


def entity_attack(args, model, model_type, tokenizer, file_in="dataset/docred/dev_wo_overlap.json"):
    attack_dir = "attack_pkl/"
    # json.dump(ori_features, open(attack_dir + model_type + '@ori_keyword_dev.pkl', 'wb'))
    tag = file_in.split("/")[-1].split(".")[0]

    en_mask_path = attack_dir + tag + "@" + model_type + "@en_mask_dev.json"
    en_shuf_path = attack_dir + tag + "@" + model_type + "@en_shuf_dev.json"
    en_repl_path = attack_dir + tag + "@" + model_type + "@en_repl_dev.json"
    ori_features = read_docred(file_in, tokenizer)
    en_mask_features = read_docred(en_mask_path, tokenizer)
    en_shuf_features = read_docred(en_shuf_path, tokenizer)
    en_repl_features = read_docred(en_repl_path, tokenizer)

    # bert on dev_wo_overlap
    all_feas = [ori_features, en_mask_features, en_shuf_features, en_repl_features]
    f1_outs = []
    args.dev_file = "dev_wo_overlap.json"
    for fea in all_feas:
        _, f1_out = evaluate(args, model, fea, tag="dev")
        print(f1_out)
        f1_outs.append(f1_out)
    en_attack_strs = ["original", "entity mask", "entity move", "entity replace"]
    for i in range(0, len(f1_outs)):
        print(f'& {en_attack_strs[i]} & {f1_outs[i]["dev_F1"]} \\\\ ')


def metric_keyword(args, model, features):
    pred = report(args, model, features)
    print(len(pred))
    spreds = sorted(pred, key=lambda x: x["title"])
    spreds = [{"title": s["title"], "r": docred_rel2id[s["r"]]} for s in spreds]
    key_res = defaultdict(list)
    for s in spreds:
        key_res[s["title"]].append(s["r"])
    all_rnum = len(features)
    pos_rnum = len(key_res)
    no_rnum = all_rnum - pos_rnum
    true_rnum = 0
    for title, rl in key_res.items():
        truth = int(title.split("_")[-1])
        if truth in rl:
            true_rnum += 1

    wrong_rnum = pos_rnum - true_rnum
    return key_res, all_rnum, pos_rnum, no_rnum, true_rnum, wrong_rnum


def attack_ratio(ori_key_res, key_res, ori_features):
    # ori no rel, now rel
    all_titles = set([f["title"] for f in ori_features])
    ori_titles = set(ori_key_res.keys())
    titles = set(key_res.keys())
    no_titles = all_titles - ori_titles
    nor_r_ratio = len(no_titles & titles) / len(no_titles)
    # ori rel, now no rel
    r_nor_ratio = len((all_titles - titles) & ori_titles) / len(ori_titles)

    no_num, true_num, false_num = 0, 0, 0
    for key in ori_key_res.keys():
        if key not in key_res:
            no_num += 1
        elif len(set(key_res[key]) & set(ori_key_res[key])) > 0:
            true_num += 1
        else:
            false_num += 1
    # ori one rel, now another rel
    rel_arel_ratio = false_num / len(ori_titles)
    # ori one rel, now rel covers
    rel_srel_ratio = true_num / len(ori_titles)
    return r_nor_ratio, nor_r_ratio, rel_arel_ratio, rel_srel_ratio


def keyword_attack(args, model, tokenizer, file_in="dataset/docred/dev_keys_new.json", limit=False):
    # create [MASK] for specific relation
    i_line, pos_samples = 0, 0
    ori_features, mask_features, anto_features, syno_features = [], [], [], []
    ori_anto_features, ori_syno_features = [], []
    mask_num, anto_num, syno_num, ori_num = 0, 0, 0, 0
    with open(file_in, "r") as fh:
        data = json.load(fh)
    kdict = pickle.load(open("dataset/docred/keywords_dict.pkl", "rb"))
    if limit:
        data = data[:10]
    entity_type = ["-", "ORG", "-", "LOC", "-", "TIME", "-", "PER", "-", "MISC", "-", "NUM"]
    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        entities = sample["vertexSet"]
        entity_start, entity_end = [], []
        mention_types = []
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
                mention_types.append(mention["type"])
        for i_s, sent in enumerate(sample["sents"]):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    t = entity_start.index((i_s, i_t))
                    mention_type = mention_types[t]
                    special_token_i = entity_type.index(mention_type)
                    special_token = ["[unused" + str(special_token_i) + "]"]
                    tokens_wordpiece = special_token + tokens_wordpiece
                    # tokens_wordpiece = ["[unused0]"]+ tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    t = entity_end.index((i_s, i_t))
                    mention_type = mention_types[t]
                    special_token_i = entity_type.index(mention_type) + 50
                    special_token = ["[unused" + str(special_token_i) + "]"]
                    tokens_wordpiece = tokens_wordpiece + special_token
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
            pos_samples += 1
        key_dict = defaultdict(list)
        for h, t, r, sent_id, st, ed, name in kdict[sample["title"]]:
            key_dict[(h, t, r)].append((sent_id, st, ed, name))
        for h, t, r in key_dict.keys():
            if [h, t] not in hts:
                continue
            mask_sents = sents.copy()
            anto_sents = []
            syno_sents = []
            anto_index_now = 0
            syno_index_now = 0
            anto_edit_len_list = [0] * len(sents)
            syno_edit_len_list = [0] * len(sents)
            has_anto, has_syno = False, False

            key_dict_sort = sorted(key_dict[(h, t, r)], key=lambda x: x[0])
            for sent_id, st, ed, name in key_dict_sort:
                # do masking strategy
                start, end = sent_map[sent_id][st], sent_map[sent_id][ed]
                for p in range(start, end):
                    mask_sents[p] = tokenizer.mask_token
                # do wordnet synonyms and antonym replace strategy
                antonyms = []
                synonyms = []
                if name is None:
                    continue
                name = name.split()
                # only consider first word
                for syn in wn.synsets(name[0]):
                    for l in syn.lemmas():
                        synonyms.append(l.name())
                        if l.antonyms():
                            antonyms.append(l.antonyms()[0].name())
                # antonyms
                if len(antonyms) > 0:
                    antonym = antonyms[0]  # first antonym
                    if antonym != name:
                        #                 print(f'anto: {antonym}, ori: {name}')
                        ori_first_token_piece = tokenizer.tokenize(name[0])
                        ori_token_len = len(ori_first_token_piece)
                        anto_token_piece = tokenizer.tokenize(antonym)
                        edit_len = len(anto_token_piece) - ori_token_len
                        # acuumulate edit len for tokens behind end
                        for i in range(len(anto_edit_len_list)):
                            if i >= end:
                                anto_edit_len_list[i] += edit_len
                        anto_sents.extend(sents[anto_index_now:start].copy())
                        anto_sents.extend(anto_token_piece)
                        end = start + ori_token_len
                        has_anto = True
                else:
                    # no antonym, retain original word
                    anto_sents.extend(sents[anto_index_now:end].copy())
                anto_index_now = end
                # synonyms
                if len(synonyms) > 0:
                    synonym = synonyms[0]  # first antonym
                    #                 print(f'anto: {antonym}, ori: {name}')
                    if synonym != name:
                        ori_first_token_piece = tokenizer.tokenize(name[0])
                        ori_token_len = len(ori_first_token_piece)
                        syno_token_piece = tokenizer.tokenize(synonym)
                        edit_len = len(syno_token_piece) - ori_token_len
                        # acuumulate edit len for tokens behind end
                        for i in range(len(syno_edit_len_list)):
                            if i >= end:
                                syno_edit_len_list[i] += edit_len
                        syno_sents.extend(sents[syno_index_now:start].copy())
                        syno_sents.extend(syno_token_piece)
                        end = start + ori_token_len
                        has_syno = True
                else:
                    # no antonym, retain original word
                    syno_sents.extend(sents[syno_index_now:end].copy())
                syno_index_now = end

            # original feature construction
            ori_sents = sents.copy()
            ori_sents = ori_sents[: MAX_SEQ_LENGTH - 2]
            ori_input_ids = tokenizer.convert_tokens_to_ids(ori_sents)
            ori_input_ids = tokenizer.build_inputs_with_special_tokens(ori_input_ids)
            ori_entity_pos = [entity_pos[h], entity_pos[t]]
            ori_hts = [[0, 1]]
            ori_relations = relations[hts.index([h, t])]
            ori_feature = {
                "input_ids": ori_input_ids,
                "entity_pos": ori_entity_pos,
                "labels": ori_relations,
                "hts": ori_hts,
                "title": f'{sample["title"]}_{h}_{t}_{r}',
            }
            ori_features.append(ori_feature)
            ori_num += 1

            # mask feature construction
            mask_sents = mask_sents[: MAX_SEQ_LENGTH - 2]
            mask_input_ids = tokenizer.convert_tokens_to_ids(mask_sents)
            mask_input_ids = tokenizer.build_inputs_with_special_tokens(mask_input_ids)
            mask_entity_pos = [entity_pos[h], entity_pos[t]]
            mask_hts = [[0, 1]]
            mask_relations = relations[hts.index([h, t])]

            mask_feature = {
                "input_ids": mask_input_ids,
                "entity_pos": mask_entity_pos,
                "labels": mask_relations,
                "hts": mask_hts,
                "title": f'{sample["title"]}_{h}_{t}_{r}',
            }
            mask_features.append(mask_feature)
            mask_num += 1

            # replace feature construction
            if has_anto:
                anto_sents.extend(sents[anto_index_now:])
                anto_sents = anto_sents[: MAX_SEQ_LENGTH - 2]
                anto_input_ids = tokenizer.convert_tokens_to_ids(anto_sents)
                anto_input_ids = tokenizer.build_inputs_with_special_tokens(anto_input_ids)
                anto_entity_pos = [
                    [(s + anto_edit_len_list[s], e + anto_edit_len_list[e]) for s, e in entity_pos[h]],
                    [(s + anto_edit_len_list[s], e + anto_edit_len_list[e]) for s, e in entity_pos[t]],
                ]
                anto_hts = [[0, 1]]
                anto_relations = relations[hts.index([h, t])]
                anto_feature = {
                    "input_ids": anto_input_ids,
                    "entity_pos": anto_entity_pos,
                    "labels": anto_relations,
                    "hts": anto_hts,
                    "title": f'{sample["title"]}_{h}_{t}_{r}',
                }
                anto_features.append(anto_feature)
                ori_anto_features.append(ori_feature)
                anto_num += 1

            if has_syno:
                syno_sents.extend(sents[syno_index_now:])
                syno_sents = syno_sents[: MAX_SEQ_LENGTH - 2]
                syno_input_ids = tokenizer.convert_tokens_to_ids(syno_sents)
                syno_input_ids = tokenizer.build_inputs_with_special_tokens(syno_input_ids)
                syno_entity_pos = [
                    [(s + syno_edit_len_list[s], e + syno_edit_len_list[e]) for s, e in entity_pos[h]],
                    [(s + syno_edit_len_list[s], e + syno_edit_len_list[e]) for s, e in entity_pos[t]],
                ]
                syno_hts = [[0, 1]]
                syno_relations = relations[hts.index([h, t])]
                syno_feature = {
                    "input_ids": syno_input_ids,
                    "entity_pos": syno_entity_pos,
                    "labels": syno_relations,
                    "hts": syno_hts,
                    "title": f'{sample["title"]}_{h}_{t}_{r}',
                }
                syno_features.append(syno_feature)
                ori_syno_features.append(ori_feature)
                syno_num += 1

    print("# of documents {}.".format(i_line))
    print("# of original num {}.".format(ori_num))
    print("# of mask num {}.".format(mask_num))
    print("# of antonym num {}.".format(anto_num))
    print("# of synonym num {}. ref count {}".format(syno_num, len(ori_syno_features)))

    # Attack ratio newer version (only consider keywords that will changed)
    print("\nstart calculating attack ratio...")
    fs = [ori_features, mask_features, ori_anto_features, anto_features, ori_syno_features, syno_features]
    key_res_list = []
    for f in fs:
        key_res, all_rnum, pos_rnum, no_rnum, true_rnum, wrong_rnum = metric_keyword(args, model, f)
        key_res_list.append(key_res)
        print(all_rnum, pos_rnum, no_rnum, true_rnum, wrong_rnum)

    fea_strs = ["Mask", "Antonym", "Synonym"]
    print("latex format: ")
    for i in range(0, len(fs), 2):
        print(
            "& %s & %.8f & %.8f & %.8f & %.8f \\\\"
            % ((fea_strs[i // 2],) + attack_ratio(key_res_list[i], key_res_list[i + 1], fs[i]))
        )  # output as latex table format
