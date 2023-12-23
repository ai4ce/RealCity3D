python pospred_vis.py --device 'cuda:0' \
    --trans_deep 6 --trans_deep_decoder 3 --num_heads 8 --embed_dim 256 --decoder_embed_dim 32 \
    --drop_ratio 0.0 --pos_weight 100 --save_dir './result_0.2/' \
    --model_dir '/home/rl4citygen/DRL4CityGen/MAGEPOLYAUTO/pospred/model/patchify_0.2_6_3_8_256_32.pth' \
    --data_path '/home/rl4citygen/DRL4CityGen/results/statespoly/poly_np.npy' \
    --datapos_path '/home/rl4citygen/DRL4CityGen/results/statespoly/polypos_np.npy' \
    --datainfo_path '/home/rl4citygen/DRL4CityGen/results/statespoly/polyinfo_np.npy'