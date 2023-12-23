python main_poly_fix.py --lr 1e-3 --device 'cuda'\
    --batch_size 64 --epochs 800 --trans_deep 12 --trans_deep_decoder 8 --num_heads 8 --save_freq 200 --embed_dim 512 --decoder_embed_dim 256 \
    --drop_ratio 0.1 --log_dir '../results/train/polyauto/output_log/64_800_12_8_8_512_256_0.1_pos20_100' \
    --output_dir '../results/train/polyauto/output_dir/64_800_12_8_8_512_256_0.1_pos20_100' --data_path '../datasets/statespoly/poly_np.npy' \
    --datapos_path '../datasets/statespoly/polypos_np.npy' --datainfo_path '../datasets/statespoly/polyinfo_np.npy'