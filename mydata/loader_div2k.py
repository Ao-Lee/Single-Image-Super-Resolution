try:
    from .dataset import MyDataset
    from .utils import InvertedProcess, ShowImage
except ImportError:
    from dataset import MyDataset
    from utils import InvertedProcess, ShowImage

class DIV2K(MyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

if __name__=='__main__':
    import sys
    sys.path.append('..')
    import cfg

    ds = DIV2K(root=cfg.PATH_DIV2K, batch_size=4, ratio=2, shape_lr=64)
    gen_tr = ds.GetGenerator_Tr()
    lr, sr = next(gen_tr)
    lr = InvertedProcess(lr)
    sr = InvertedProcess(sr)
    for i in range(4):
        low = lr[i, ...]
        high = sr[i, ...]
        ShowImage(low, figsize=(5, 5))
        ShowImage(high, figsize=(5, 5))
        