from transformers import AutoTokenizer
from run_utils import build_entity_attack_dataset, enp_topk, get_atlop_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="roberta-large", type=str)
ds_args = parser.parse_args()


BERT_DIR = "" 
# model_type = "roberta-large"  # or bert-base-cased
model_type = ds_args.model_type
tokenizer = AutoTokenizer.from_pretrained(BERT_DIR + model_type)
args, model = get_atlop_model(model_type)

enp_topk(args, model, model_type, tokenizer, "dataset/ig_pkl")
build_entity_attack_dataset(model_type, tokenizer)
