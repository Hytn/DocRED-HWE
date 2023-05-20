# DocRED_HWE
Source code for ACL 2023 paper: Did the Models Understand Documents? Benchmarking Models for Language Understanding in Document-Level Relation Extraction

> Document-level relation extraction (DocRE) attracts more research interest recently. While
> models achieve consistent performance gains in
> DocRE, their underlying decision rules are still
> understudied: Do they make the right predictions according to rationales? In this paper, we
> take the first step toward answering this question and then introduce a new perspective on
> comprehensively evaluating a model. Specifically, we first conduct annotations to provide
> the rationales considered by humans in DocRE.
> Then, we conduct investigations and reveal the
> fact that: In contrast to humans, the representative state-of-the-art (SOTA) models in DocRE
> exhibit different decision rules. Through our
> proposed RE-specific attacks, we next demonstrate that the significant discrepancy in decision rules between models and humans severely
> damages the robustness of models and renders
> them inapplicable to real-world RE scenarios.
> After that, we introduce mean average precision (MAP) to evaluate the understanding and
> reasoning capabilities of models. According to
> the extensive experimental results, we finally
> appeal to future work to consider evaluating
> both performance and the understanding ability
> of models for the development of their applications. 

## Dataset

`DocRED_HWE.json`: DocRED with Human-annotated Word-level Evidence(HWE) dataset.

Statistics of the 699 documents (the same as DocRED_HWE's) from the original validation dataset of DocRED:

1. evidence sentence number: 12000
2. relational fact number: 7342
3. document number: 699

## Corrected Annotation Errors

`annotation errors in DocRED.xls`: Annotation errors corrected by our annotators on the validation set of DocRED.

## Codes

`build_attack_dataset.ipynb`: Performing keyword attack and entity attack on DocRED_HWE.json and docred_dev.json respectively. 

`MAP_metric.ipynb`: Evaluating with MAP metric

`plot.ipynb`: Ploting MAP curves and TopK-F1 curves.

`eval_attack_docunet.ipynb`: Evaluating DocuNet's performance under two attacks.

`eval_attack_atlop.ipynb`: Evaluating ATLOP's performance under two attacks.

`run_attacks.py`: All attacks on ATLOP.

`IG_inference.py`: Calculating integrated gradient (IG) to attribute ATLOP.



## License

MIT
