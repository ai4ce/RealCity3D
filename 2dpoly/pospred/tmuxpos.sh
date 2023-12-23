python train_pospred.py --lr 1e-3 --device 'cuda:0' \
    --batch_size 4 --epochs 800 --trans_deep 6 --trans_deep_decoder 3 --num_heads 8 --save_freq 200 --embed_dim 256 --decoder_embed_dim 32 \
    --drop_ratio 0.0 --pos_weight 100 --log_dir '/home/rl4citygen/DRL4CityGen/results/polytest/output_log/8_800_6_3_8_256_64_0.0_pos100' \
    --output_dir '/home/rl4citygen/DRL4CityGen/results/polytest/output_dir/8_800_6_3_8_256_64_0.0_pos100' \
    --data_path '/home/rl4citygen/DRL4CityGen/results/statespoly/poly_np.npy' \
    --datapos_path '/home/rl4citygen/DRL4CityGen/results/statespoly/polypos_np.npy' \
    --datainfo_path '/home/rl4citygen/DRL4CityGen/results/statespoly/polyinfo_np.npy'