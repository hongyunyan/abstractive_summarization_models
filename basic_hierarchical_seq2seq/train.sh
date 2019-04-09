rm -rf ./abstractor
CUDA_VISIBLE_DEVICES=1 python abstractor_main.py --path ./abstractor --w2v word2vec/word2vec.128d.226k.bin --data_path ~/perfect_dataset

