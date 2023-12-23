python train_pospred.py --lr 1e-3 --device 'cuda:0' --patchify True\
    --batch_size 4 --epochs 800 --trans_deep 6 --trans_deep_decoder 3 --num_heads 8 --save_freq 200 --embed_dim 256 --decoder_embed_dim 32 \
    --drop_ratio 0.1 --pos_weight 100 --log_dir '../../results/train/pospred3d/output_log/4_800_6_3_8_256_32_0.1_pos100' \
    --output_dir '../../results/train/pospred3d/output_dir/4_800_6_3_8_256_32_0.1_pos100' --data_path '../../datasets/3Dpoly/poly_np.npy' \
    --datapos_path '../../datasets/3Dpoly/polypos_np.npy' --datainfo_path '../../datasets/3Dpoly/polyinfo_np.npy' --datah_path '../../datasets/3Dpoly/polyh_np.npy'