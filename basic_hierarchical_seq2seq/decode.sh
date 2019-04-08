rm -rf result
python decode_model.py --path=result --model_dir=abstractor --beam 3 --data_path dataset --test
