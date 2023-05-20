import argparse
import torch
from apex import amp
from transformers import AutoConfig, AutoModel, AutoTokenizer
from model import DocREModel
from attacks_utils import keyword_attack, build_entity_attack_dataset, entity_attack


def arg_pre_atlop():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--load_path", default="saved_dict/model_roberta.ckpt", type=str)
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels in prediction.")

    parser.add_argument(
        "--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")

    args = parser.parse_args(args=[])
    return args


def get_atlop_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = arg_pre_atlop()
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.model_name_or_path = model_type
    if model_type.startswith("r"):
        args.transformer_type = "roberta"
        args.load_path = "saved_dict/model_roberta.ckpt"
    else:
        args.transformer_type = "bert"
        args.load_path = "saved_dict/model_bert.ckpt"

    BERT_DIR = "/cpfs/user/cht/cbsp/"  # TODO: rm
    model_name_or_path = BERT_DIR + args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )

    model = AutoModel.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(0)
    model = amp.initialize(model, opt_level="O1", verbosity=0)

    model.load_state_dict(torch.load(args.load_path, map_location=device))
    print("load model from ", args.load_path)

    return args, model


def arg_pre_docunet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="roberta", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--load_path", default="checkpoint/docred/roberta.pt", type=str)
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels in prediction.")
    parser.add_argument("--unet_in_dim", type=int, default=3, help="unet_in_dim.")
    parser.add_argument("--unet_out_dim", type=int, default=256, help="unet_out_dim.")
    parser.add_argument("--down_dim", type=int, default=256, help="down_dim.")
    parser.add_argument("--channel_type", type=str, default="context-based", help="unet_out_dim.")
    parser.add_argument("--max_height", type=int, default=42, help="log.")
    parser.add_argument("--dataset", type=str, default="docred", help="dataset type")

    args = parser.parse_args(args=[])
    return args


BERT_DIR = "/cpfs/user/cht/cbsp/"  # TODO: rm path
model_type = "roberta-large"  # or bert-base-cased
tokenizer = AutoTokenizer.from_pretrained(BERT_DIR + model_type)
args, model = get_atlop_model(model_type)

keyword_attack(args, model, tokenizer)


build_entity_attack_dataset(model_type, tokenizer)
entity_attack(args, model, model_type, tokenizer)
