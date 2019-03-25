rm -rf result
python decode_full_model.py --path result --model_dir . --data_path dataset --beam 5 --test
