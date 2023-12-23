# python main_poly_fix.py --lr 1e-3 --device 'cuda'\
#     --batch_size 64 --epochs 1500 --trans_deep 6 --num_heads 8 --save_freq 400 --embed_dim 512 \
#     --drop_ratio 0.1 --log_dir '../results/train/polyclassification/output_log/64_1500_12_8_512_0.1' \
#     --output_dir '../results/train/polyclassification/output_dir/64_1500_12_8_512_0.1' --data_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/poly_np.npy' \
#     --datapos_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polypos_np.npy' --datainfo_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polyinfo_np.npy'\
#     --dataclass_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polyclass_np.npy'

python main_probing.py --lr 1e-3 --device 'cuda'\
    --batch_size 64 --epochs 1500 --trans_deep 6 --num_heads 8 --save_freq 400 --embed_dim 512 \
    --drop_ratio 0.1 --log_dir '../results/train/polyclassification/output_log/64_1500_12_8_512_0.1_probing' \
    --output_dir '../results/train/polyclassification/output_dir/64_1500_12_8_512_0.1_probing' --data_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/poly_np.npy' \
    --datapos_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polypos_np.npy' --datainfo_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polyinfo_np.npy'\
    --dataclass_path '/mnt/NAS/data/WenyuHan/LargeCity/Final/polyclass_np.npy'