python pospred_vis.py --device 'cuda' \
    --trans_deep 6 --trans_deep_decoder 3 --num_heads 8 --embed_dim 512 --decoder_embed_dim 16 \
    --drop_ratio 0.1 --pos_weight 100 --save_dir './result_road/' \
    --model_dir '/scratch/rx2281/pytorch/InfiniteCityGen/results/train/polypospredroad/output_dir/128_3000_6_3_8_512_16_0.1_pos100_1e-3/mae_reconstruction_best.pth' \
    --data_path '../../datasets/dataroad/poly_np.npy' \
    --datapos_path '../../datasets/dataroad/polypos_np.npy' \
    --datainfo_path '../../datasets/dataroad/polyinfo_np.npy' \
    --dataroad_path '../../datasets/dataroad/polyroad_np.npy'