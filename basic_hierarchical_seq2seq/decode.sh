rm -rf result
python decode_model.py --path=result --model_dir=abstractor --beam 1 --data_path ../../../perfect_dataset --test
