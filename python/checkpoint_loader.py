import numpy as np

def main():
    chk_point = np.load('vit_s16.npz')
    print(chk_point['head/kernel'].shape)
if __name__ == '__main__':
    main()