rm -rf result
python decode_model.py --path=result --model_dir=abstractor2 --beam 1 --data_path dataset --test
