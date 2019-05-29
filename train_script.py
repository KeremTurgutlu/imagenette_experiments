from fastai.script import *
from fastai.vision import *
from fastai.vision.models.wrn import wrn_22
from fastai.distributed import *
from fastai.callbacks import *
from models import *
from time import time
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
    lr:Param("learning rate", float)=1e-2,
    mixup:Param("Mixup beta", float)=None,
    epochs:Param("Number of epochs", int)=5,
    fp16:Param("fp16 or fp32", int)=0,
    label_smooth:Param("label smoothing or not", int)=0,
    arch_name:Param("Arch name", str)="xresnet18",
    alpha_pool:Param("Alpha pool or concat pool", int)=0,
    logdir:Param("Alpha pool or concat pool", str)='exp1'
    ):
    """
    Single: python train_scipt.py --gpu=0 
    Distributed: python fastai/launch.py --gpus=1234567 train_script.py
    """
    n_gpus = num_distrib()
    if n_gpus>0: gpu = setup_distrib(gpu)
    workers = min(16, num_cpus()//n_gpus) if n_gpus>0 else num_cpus()

    # data
    data = get_data(size, woof, bs, workers=workers)
    
    # callbacks, log to distinct dir
    experiment_str = f"""
    size: {size}
    woof: {woof}
    bs: {bs}
    lr: {lr}
    mixup: {mixup}
    epochs: {epochs}
    fp16: {fp16}
    label_smooth: {label_smooth}
    arch_name: {arch_name}
    alpha_pool: {alpha_pool}
    """

    learn_callbacks = [TerminateOnNaNCallback()]
    learn_callback_fns = [partial(CSVLogger, filename=f'./logs/{logdir}/{str(int(time()))}')]

    # model
    arch = arch_dict[arch_name]
    learn = cnn_learner(data=data, 
                        custom_head=custom_head if alpha_pool else None,
                        base_arch=arch,
                        pretrained=False)
    m = learn.model; learn.destroy()
    
    # learn
    opt_func = partial(optim.Adam, betas=(0.9,0.99), eps=1e-6)
    learn = Learner(data, m, wd=1e-2, opt_func=opt_func,
            metrics=[accuracy], bn_wd=False, true_wd=True,
            loss_func = LabelSmoothingCrossEntropy() if label_smooth else CrossEntropyFlat(),
            callbacks=learn_callbacks,
            callback_fns=learn_callback_fns)

    if n_gpus>0:
        if gpu is None: learn.model = nn.DataParallel(learn.model)
        else: learn.to_distributed(gpu)

    if fp16: learn.to_fp16()
    if mixup: learn.mixup(mixup)
    
    # fit
    learn.fit_one_cycle(epochs, lr, div_factor=10, pct_start=0.3)

    # save experiment details        
    with open(f'{learn.path}/./logs/{logdir}/experiment.txt', 'w') as f:
        f.write(experiment_str)
