import numpy as np


if __name__=="__main__":
    datas = np.load('../results/states/poly_np_norm.npy')
    # datas1 = np.load('../results/states/polyinfo_np.npy')
    # datas2 = np.load('../results/states/polypos_np.npy')
   
    
    np.save(f"../results/states/poly_np_norm_100.npy", datas[:100])
    # np.save(f"../results/states/polyinfo_np_100.npy", datas1[:100])
    # np.save(f"../results/states/polypos_np_100.npy", datas2[:100])