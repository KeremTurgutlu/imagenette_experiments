from fastai.vision import *
from xresnet2 import *

__all__ = ['arch_dict', 'custom_head']

act_fn = nn.ReLU(inplace=True)

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

# Inspired by https://arxiv.org/pdf/1805.08318.pdf, https://github.com/sdoria/SimpleSelfAttention
class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in:int, ks=1):#, n_out:int):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)
        self.gamma = nn.Parameter(tensor([0.]))

    def forward(self,x):
        size = x.size()
        x = x.view(*size[:2],-1)
        o = torch.bmm(x.permute(0,2,1).contiguous(),self.conv(x))
        o = self.gamma * torch.bmm(x,o) + x
        return o.view(*size).contiguous()

# Modified from https://github.com/fastai/fastai/blob/9b9014b8967186dc70c65ca7dcddca1a1232d99d/fastai/vision/models/xresnet.py
# Added self attention
class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1,sa=False):
        super().__init__()
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        
        self.sa = SimpleSelfAttention(nf,ks=1) if sa else noop
        self.convs = nn.Sequential(*layers)
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): 
        return act_fn(self.sa(self.convs(x)) + self.idconv(self.pool(x)))

class XResNet_sa(nn.Sequential):
    @classmethod
    def create(cls, expansion, layers, c_in=3, c_out=1000):
        nfs = [c_in, (c_in+1)*8, 64, 64]
        stem = [conv_layer(nfs[i], nfs[i+1], stride=2 if i==0 else 1)
            for i in range(3)]

        nfs = [64//expansion,64,128,256,512]
        res_layers = [cls._make_layer(expansion, nfs[i], nfs[i+1],
                                      n_blocks=l, stride=1 if i==0 else 2, sa = True if i in [len(layers)-4] else False)
                  for i,l in enumerate(layers)]
        res = cls(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *res_layers,
            
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(nfs[-1]*expansion, c_out),
        )
        init_cnn(res)
        return res

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, sa = False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1, sa if i in [n_blocks -1] else False)
              for i in range(n_blocks)])
    
def xresnet18_sa(pretrained=False, **kwargs):
    model = XResNet_sa.create(1, [2, 2, 2, 2], **kwargs)
    if pretrained: pass
    return model

def xresnet34_sa(pretrained=False, **kwargs):
    model = XResNet_sa.create(1, [3, 4, 6 ,3], **kwargs)
    if pretrained: pass
    return model

def xresnet50_sa(pretrained=False, **kwargs):
    model = XResNet_sa.create(4, [3, 4, 6, 3], **kwargs)
    if pretrained: pass
    return model

class AlphaPool(nn.Module):
    def __init__(self, alpha:float=1., eps:float=1e-8):
        super().__init__()
        self.alpha = nn.Parameter(tensor([0.]))   
        self.eps = eps
        
    def forward(self, x): 
        "Creates alpha-pooling features from a CNN like feature map"
        self.alpha.data.sigmoid_()
        b,fn,h,w = x.shape
        x = x.view(b,fn,h*w)
        x1 = torch.sign(x)*torch.sqrt(((torch.abs(x) + 1e-5)**(self.alpha)))
        x1 = x1.permute(0,2,1).contiguous().unsqueeze(2)
        x2 = x.permute(0,2,1).contiguous().unsqueeze(3)
        x = (x1*x2).view(b,h*w,-1)
        x = F.normalize(x.mean(dim=1))
        return x

def create_head(nf:int, nc:int, lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5,
                concat_pool:bool=True,alpha_pool:bool=True, bn_final:bool=False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    pool = AlphaPool() if alpha_pool else pool
    layers = [pool, Flatten()]
    for ni,no,p,actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)

custom_head = create_head(512**2, 10)

arch_dict = {
    'resnet18':models.resnet18,
    'resnet34':models.resnet34,
    'xresnet18':xresnet18,
    'xresnet34':xresnet34_2,
    'xresnet18_sa':xresnet18_sa,
    'xresnet34_sa':xresnet34_sa,
}