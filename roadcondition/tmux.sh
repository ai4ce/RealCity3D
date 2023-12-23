python main_poly_road.py --lr 1e-3 --max_build 250 --discre 100 --remain_num 30\
			--batch_size 64 --epochs 1000 --trans_deep 12 --trans_deep_decoder 8 --num_heads 8 --save_freq 200 --embed_dim 512 --decoder_embed_dim 512 \
			--drop_ratio 0.1 --log_dir '../results/train/polyautoroad/output_log/64_1000_12_8_8_512_512_0.1_1e-3' \
			--output_dir '../results/train/polyautoroad/output_dir/64_1000_12_8_8_512_512_0.1_1e-3' --data_path '../datasets/dataroad/poly_np.npy' \
    		--datapos_path '../datasets/dataroad/polypos_np.npy' --datainfo_path '../datasets/dataroad/polyinfo_np.npy' --dataroad_path '../datasets/dataroad/polyroad_np.npy'