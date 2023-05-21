# DocRED-HWE

Source code for ACL 2023 paper: Did the Models Understand Documents? Benchmarking Models for Language Understanding in Document-Level Relation Extraction

> Document-level relation extraction (DocRE) attracts more research interest recently. While models achieve consistent performance gains in DocRE, their underlying decision rules are still understudied: Do they make the right predictions according to rationales?
>
> In this paper, we take the first step toward answering this question and then introduce a new perspective on
> comprehensively evaluating a model. Specifically, we first conduct annotations to provide the rationales considered by humans in DocRE.Then, we conduct investigations and reveal the fact that: In contrast to humans, the representative state-of-the-art (SOTA) models in DocRE exhibit different decision rules. Through our proposed RE-specific attacks, we next demonstrate that the significant discrepancy in decision rules between models and humans severely damages the robustness of models and renders them inapplicable to real-world RE scenarios. After that, we introduce mean average precision (MAP) to evaluate the understanding and reasoning capabilities of models.
>
> According to the extensive experimental results, we finally appeal to future work to consider evaluating both performance and the understanding ability of models for the development of their applications.

## Dataset

`DocRED_HWE.json`: DocRED with Human-annotated Word-level Evidence(HWE) dataset.

Statistics of the 699 documents (the same as DocRED_HWE's) from the original validation dataset of DocRED:

1. evidence sentence number: 12000
2. relational fact number: 7342
3. document number: 699

## Corrected Annotation Errors

`annotation errors in DocRED.xls`: Annotation errors corrected by our annotators on the validation set of DocRED.

## Codes

- `MAP_metric.ipynb`: Evaluating with MAP metric
- `plot.ipynb`: Ploting MAP curves and TopK-F1 curves.
- `eval_attack_docunet.ipynb`: Evaluating DocuNet's performance under two attacks.
- `MAP_metric.py`: evaluate model with MAP (mean average precision)
- `IG_inference.py`: Calculating integrated gradient (IG) to attribute ATLOP.
- `get_ds.py`: Generate dataset for evalution.
- `run_attacks.py`: All attacks on ATLOP.

## Dependencies

#### Using pip

```shell
pip install -r requirements.pip.txt
```

#### Using conda

```shell
conda install --file requirements.conda.txt
```

- install in a conda virtual environment

## Preparation

**Step1**. Prepare original ATLOP trained model, saved to `saved_dict/`, name it `saved_dict/model_bert.ckpt` or `saved_dict/model_roberta.ckpt`

**Step2**. Use IG to generate the weights of every token for specific relation fact

```shell
python IG_inference.py --infer_method INFER_METHOD --load_path LOAD_PATH --model_name_or_path MODEL_NAME_OR_PATH --transformer_type TRANSFORMER_TYPE
```

- `INFER_METHOD` is the attribution method, use `ig_infer` or `grad_infer`, `LOAD_PATH` is your saved model checkpoint, `MODEL_NAME_OR_PATH` and `TRANSFORMER_TYPE` are BERT's parameter, you can set to `roberta-large` and `roberta`, respectively.
- result of IG will be saved to `dataset/ig_pkl` folder

**Step3**. Generate ENP_TOPK dataset(entity pair with topk attributed tokens), and entity name attack dataset (three types mentioned in paper)

```shell
python getds.py --model_type MODEL_TYPE
```

- `MODEL_TYPE` is your saved model type, should be `roberta-large` or `bert-base-cased`
- ENP_TOPK dataset and entity name attack dataset will be stored in `dataset/docred/enp_topk/` and `dataset/attack_pkl/` folder, respectively

## Evaluation

### Run MAP evaluation

```shell
python MAP_metric.py --model_type MODEL_TYPE
```

- use IG result to generate the new MAP evaluation, output will be saved to `dataset/keyword_pkl/`, which you can draw the line chart like in `plot.ipynb`

### Run word-level evidence attack and entity name attack on the generated dataset

```shell
python run_attack.py --model_type MODEL_TYPE
```

- run the word-level evidence attack and entity name attack, output will be printed in STDOUT

## Contacts

If you have any questions, please contact [Haotian Chen](mailto:htchen18@fudan.edu.cn), we will reply it as soon as possible.

## License

MIT
