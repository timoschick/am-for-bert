# WordNet Language Model Probing

This repository contains the **WordNet Language Model Probing** (WNLaMPro) dataset. Each line of the dataset file (`dataset/WNLaMPro.txt`) has the following form (note that all columns are separated by tabs rather than spaces):

    <ID>  <SET_TYPE>  <KEY_WORD>  <RELATION>  <TARGET_WORD1>  <TARGET_WORD2>  ...
    
The columns have the following meaning:

- `<ID>`: A unique identifier for this dataset entry
- `<SET_TYPE>`: Either `test` or `dev`, depending on whether this entry belongs to the development or test subset of PSR
- `<KEY_WORD>`: The key word in the `<ANNOTATED_WORD>` format (see below)
- `<RELATION>`: The relation of this entry, either `antonym`, `hypernym`, `cohyponym` or `corruption`
- `<TARGET_WORDn>`: The `n`-th target word for this dataset entry, in the `<ANNOTATED_WORD>` format (see below)

### Annotated Words

Each key and target word of the WNLaMPro dataset is represented as an `<ANNOTATED_WORD>` in the following form:

    <ANNOTATED_WORD> := <WORD> (<POS>,<FREQ>,<COUNT>)
    
The columns have the following meaning:

- `<WORD>`: The actual word
- `<POS>`: The part-of-speech tag for this word (either `n`oun or `a`djective)
- `<FREQ>`: The estimated Zipf frequency for this word, obtained using [wordfreq](https://pypi.org/project/wordfreq/)
- `<COUNT>`: The number of occurrences of this word in the [Westbury Wikipedia corpus](http://www.psych.ualberta.ca/~westburylab/downloads/westburylab.wikicorp.download.html)

## Evaluation Script

You can evaluate a pretrained language model on WNLaMPro as follows:
```
python3 eval-script/evaluate.py --root ROOT --predictions_file PREDICTIONS_FILE --output_file OUTPUT_FILE --model_cls MODEL_CLS --model_name MODEL_NAME (--embeddings EMBEDDINGS)
```
where
- `ROOT` is the path to the directory where `WNLaMPro.txt` can be found;
- `PREDICTIONS_FILE` is the name of the file in which predictions are to be stored (relative to `ROOT`);
- `OUTPUT_FILE` is the name of the file in which the model's MRR is to be stored (relative to `ROOT`);
- `MODEL_CLS` is either `bert` or `roberta` (the evaluation script currently does not support other pretrained language models);
- `MODEL_NAME` is either the name of a pretrained model from the [Hugging Face Transformers Library](https://github.com/huggingface/transformers) (e.g., `bert-base-uncased`) or the path to a finetuned model;
- `EMBEDDINGS` (optional) is the path (relative to `ROOT`) of a file that contains embeddings which are used to overwrite the language model's original embeddings. Each line of this file has to be in the format `<WORD> <EMBEDDING>`, for example `apple -0.12 3.45 0.23 ... 0.03`.

For additional parameters, check the content of `eval-script/evaluate.py` or run `python3 eval-script/evaluate.py --help`. 

## Citation

If you make use of the WNLaMPro dataset, please cite the following paper:

```
@inproceedings{schick2020rare,
  title={Rare words: A major problem for contextualized representation and how to fix it by attentive mimicking},
  author={Schick, Timo and Sch{\"u}tze, Hinrich},
  url="https://arxiv.org/abs/1904.06707",
  booktitle={Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
