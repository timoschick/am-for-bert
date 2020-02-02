# WordNet Language Model Probing

This repository contains the **WordNet Language Model Probing** (WNLaMPro) dataset. Each line of the dataset has the following form (note that all columns are separated by tabs rather than spaces):

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

Documentation for the evaluation script will soon be added.