# Open-Domain-Question-Answering
code for QA project 21-22
- [ ] baseline model BM25 + BERT
- [ ] ORQA
- [ ] RAG

starter code 
- working with Open SQuAD and NQA datasets
- download SQuAD from offical site and run `download_squad.py` to format it accordingly, currently ignores all qa pairs which have tag `is_impossible= True` 
- `data_utils.py` has the common dataloader for any QA dataset
- `model.py` and `train.py` base code for Training T5 in a closed book setting
