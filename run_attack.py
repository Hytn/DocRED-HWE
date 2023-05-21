from transformers import AutoTokenizer
import argparse
from run_utils import get_atlop_model, keyword_attack, entity_attack

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="roberta-large", type=str)
att_args = parser.parse_args()

BERT_DIR = ""  # local m path
model_type = att_args.model_type
tokenizer = AutoTokenizer.from_pretrained(BERT_DIR + model_type)
args, model = get_atlop_model(model_type)

keyword_attack(args, model, tokenizer)

entity_attack(args, model, model_type, tokenizer)
