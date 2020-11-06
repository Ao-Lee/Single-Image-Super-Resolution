from mydata.dataset import MyDataset
from mydata.transforms import MyRandomCrop, ToNumpy, MyCompose

class COCO(MyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def GetShapeLow(self):
        return (64, 64, 3)
        
    def GetTransformFnTr(self):
        size = self.GetShapeLow()[:-1]
        operations = []
        operations.append(MyRandomCrop(size=size, ratio=self.ratio))
        operations.append(ToNumpy())
        return MyCompose(operations)
    
    def GetTransformFnVal(self):
        size = self.GetShapeLow()[:-1]
        operations = []
        operations.append(MyRandomCrop(size=size, ratio=self.ratio))
        operations.append(ToNumpy())
        return MyCompose(operations)
    
if __name__=='__main__':
    import cfg
    ds = COCO(root=cfg.PATH_COCO, ratio=2, batch_size=4)
    ds._Test()