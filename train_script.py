from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
torch.backends.cudnn.benchmark = True

# data 
def get_data(size, woof, bs, workers=None):
    if   size<=128: path = URLs.IMAGEWOOF_160 if woof else URLs.IMAGENETTE_160
    elif size<=224: path = URLs.IMAGEWOOF_320 if woof else URLs.IMAGENETTE_320
    else          : path = URLs.IMAGEWOOF     if woof else URLs.IMAGENETTE
    path = untar_data(path)

    n_gpus = num_distrib() or 1
    if workers is None: workers = min(8, num_cpus()//n_gpus)

    return (ImageList.from_folder(path).split_by_folder(valid='val')
            .label_from_folder().transform(([flip_lr(p=0.5)], []), size=size)
            .databunch(bs=bs, num_workers=workers)
            .presize(size, scale=(0.35,1))
            .normalize(imagenet_stats))

@call_parse
def main(
    gpu:Param("GPU to run on", str)=None,
    size:Param("Image size", int)=128,
    woof:Param("Imagewoof or Imageneette", int)=1,
    bs:Param("Batch size", int)=64,
    mixup:Param("Mixup beta", float)=None,
    epochs:Param("Number of epochs", int)=5,
    fp16:Param("fp16 or fp32", int)=0,
     ):
    """
    Single: python -gpu=0 train_scipt.py
    Distributed: python fastai/launch.py --gpus=1234567 train_script.py
    """
    gpu = setup_distrib(gpu)
    n_gpus = num_distrib()
    workers = min(16, num_cpus()//n_gpus)

    # data
    get_data(size, woof, bs, workers=workers)

    # learn
    learn = (Learner(data, m, wd=1e-2, opt_func=opt_func, metrics=[accuracy],
             bn_wd=False, true_wd=True, loss_func = LabelSmoothingCrossEntropy()))
    
    if gpu is None: learn.model = nn.DataParallel(learn.model)
    else: learn.to_distributed(gpu)

    if fp16: learn.to_fp16()
    if mixup: learn.mixup(mixup)
    
    # fit
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)
        