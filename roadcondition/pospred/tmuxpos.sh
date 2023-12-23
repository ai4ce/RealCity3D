python train_pospred.py --lr 1e-3 --device 'cuda' \
    --batch_size 4 --epochs 800 --trans_deep 6 --trans_deep_decoder 3 --num_heads 8 --save_freq 200 --embed_dim 256 --decoder_embed_dim 16 \
    --drop_ratio 0.1 --pos_weight 100 --log_dir '../../results/train/road/output_log/8_800_6_3_8_256_64_0.0_pos100' \
    --output_dir '../../results/train/road/output_dir/8_800_6_3_8_256_64_0.0_pos100' \
    --data_path '../../datasets/dataroad/poly_np.npy' \
    --datapos_path '../../datasets/dataroad/polypos_np.npy' --datainfo_path '../../datasets/dataroad/polyinfo_np.npy' \
    --dataroad_path '../../datasets/dataroad/polyroad_np.npy'