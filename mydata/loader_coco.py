try:
    from .dataset import MyDataset
    from .utils import InvertedProcess, ShowImage
except ImportError:
    from dataset import MyDataset
    from utils import InvertedProcess, ShowImage

class COCO(MyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

if __name__=='__main__':
    import sys
    sys.path.append('..')
    import cfg
    from tqdm import tqdm

    batch_size = 10
    # ds = COCO(root=cfg.PATH_COCO, batch_size=batch_size, ratio=2, shape_lr=64)
    ds = COCO(root=cfg.PATH_COCO, ratio=2, debug=True)
    gen_tr = ds.GetGenerator_Tr()
    lr, sr = next(gen_tr)
    lr = InvertedProcess(lr)
    sr = InvertedProcess(sr)
    
    low = lr[0, ...]
    high = sr[0, ...]
    #ShowImage(low, figsize=(5, 5))
    #ShowImage(high, figsize=(5, 5))
        
    for _ in tqdm(range(1000)):
        lr, sr = next(gen_tr)
            