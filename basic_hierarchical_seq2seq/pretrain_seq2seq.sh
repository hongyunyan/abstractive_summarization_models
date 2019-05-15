rm -rf pretrain_seq2seq
python pretrain_seq2seq.py --pretrain_path ./pretrain_abstractor --path ./pretrain_seq2seq --w2v word2vec/word2vec.128d.226k.bin --data_path ~/perfect_dataset 
