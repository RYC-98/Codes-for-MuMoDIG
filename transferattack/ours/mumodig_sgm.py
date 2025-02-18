import torch
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
import torch.nn as nn

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from ..lb_quantization import LBQuantization

import scipy.stats as st
# import random

class MUMODIG_SGM(MIFGSM):

    # 注意测试时是 num_scale=5
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., 
                 N_trans = 6, N_base = 1, N_intepolate = 1, region_num = 2, lamb = 0.65, # 6,1,1,2,0.65
                 targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None, attack='EPIG', **kwargs):

        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack) 
        self.N_trans = N_trans    
        self.N_base = N_base
        self.N_intepolate = N_intepolate    
        self.quant = LBQuantization(region_num)
        self.lamb = lamb

        register_hook_for_resnet(self.model, arch=model_name, gamma=0.2)


    def forward(self, data, label, **kwargs):
        """
        The general attack procedure

        Arguments:
            data (N, C, H, W): tensor for input images
            labels (N,): tensor for ground-truth labels if untargetd
            labels (2,N): tensor for [ground-truth, targeted labels] if targeted
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0


        for iter_out in range(self.epoch):

            
            sole_grad = self.ig(data, delta, label)
            exp_grad = self.exp_ig(data, delta, label)
            ig_grad = sole_grad  + exp_grad

            momentum = self.get_momentum(ig_grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)


        return delta.detach()


    def ig(self, data, delta, label, **kwargs): 
        
        ig = 0

        for i_base in range(self.N_base):

            baseline = self.quant(data+delta).clone().detach().to(self.device) 


            path = data+delta - baseline
            acc_grad = 0   
            for i_inter in range(self.N_intepolate):

                x_interplotate = baseline + (i_inter + self.lamb) * path / self.N_intepolate 
                logits = self.get_logits(x_interplotate)

                loss = self.get_loss(logits, label)
                
                if i_base + 1  == self.N_base and i_inter + 1 == self.N_intepolate:
                    each_ig_grad = self.get_grad(loss, delta)
                else:
                    each_ig_grad = self.get_repeat_grad(loss, delta) # 
                    
                acc_grad += each_ig_grad 


            ig += acc_grad * path 


        return ig


    def exp_ig(self, data, delta, label, **kwargs):
        
        ig = 0

        for i_trans in range(self.N_trans):

            x_transform = self.select_transform_apply(data+delta)

            for i_base in range(self.N_base):

                baseline = self.quant(x_transform).clone().detach().to(self.device) # quant baseline

                path = x_transform - baseline 
                
                acc_grad = 0          
                for i_inter in range(self.N_intepolate):

                    x_interplotate = baseline + (i_inter + self.lamb) / self.N_intepolate * path  

                    logits = self.get_logits(x_interplotate)

                    loss = self.get_loss(logits, label)

                    if i_base + 1  == self.N_base and i_inter + 1 == self.N_intepolate:
                        each_ig_grad = self.get_grad(loss, delta)
                    else:
                        each_ig_grad = self.get_repeat_grad(loss, delta) # 

                    acc_grad += each_ig_grad 

                ig += acc_grad * path  
 


        return ig


    def get_repeat_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=True, create_graph=False)[0]


    def vertical_shift(self, x):
        _, _, w, _ = x.shape         
        step = np.random.randint(low = 0, high=w, dtype=np.int32) # w = 224
        return x.roll(step, dims=2)   # step 是滚动的步数, dim 是滚动的维度

    def horizontal_shift(self, x):
        _, _, _, h = x.shape          
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))     

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def random_rotate(self, x):
        rotation_transform = K.RandomRotation(p =1, degrees=45)
        return rotation_transform(x)
    
    def random_affine(self, x):
        trans_list = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.random_rotate]

        i = torch.randint(0, len(trans_list), [1]).item()
        trans = trans_list[i]
        return trans(x)

    def random_resize_and_pad(self, x, img_large_size = 245, **kwargs):

        img_inter_size = torch.randint(low=min(x.shape[-1], img_large_size), high=max(x.shape[-1], img_large_size), size=(1,), dtype=torch.int32)
        img_inter = F.interpolate(x, size=[img_inter_size, img_inter_size], mode='bilinear', align_corners=False)
        res_space = img_large_size - img_inter_size
        res_top = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_bottom = res_space - res_top
        res_left = torch.randint(low=0, high=res_space.item(), size=(1,), dtype=torch.int32)
        res_right = res_space - res_left
        padded = F.pad(img_inter, [res_left.item(), res_right.item(), res_top.item(), res_bottom.item()], value=0)
        x_trans = F.interpolate(padded, size=[x.shape[-1], x.shape[-1]], mode='bilinear', align_corners=False)
        return x_trans
    

    def select_transform_apply(self, x, **kwargs):

        T_set = [self.random_affine, self.random_resize_and_pad] # 

        i = torch.randint(0, len(T_set), [1]).item()
        trans = T_set[i]

        return trans(x)



def backward_hook(gamma):
    """
    implement SGM through grad through ReLU
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    def _backward_hook(module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (gamma * grad_in[0],)
    return _backward_hook


def backward_hook_norm(module, grad_in, grad_out):
    """
    normalize the gradient to avoid gradient explosion or vanish
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    std = torch.std(grad_in[0])
    return (grad_in[0] / std,)


def register_hook_for_resnet(model, arch, gamma):
    """
    register hook for resnet models
    (This code is copied from https://github.com/csdongxian/skip-connections-matter)
    """
    # There is only 1 ReLU in Conv module of ResNet-18/34
    # and 2 ReLU in Conv module ResNet-50/101/152
    if arch in ['resnet50', 'resnet101', 'resnet152']:
        gamma = np.power(gamma, 0.5)
    backward_hook_sgm = backward_hook(gamma)

    for name, module in model.named_modules():
        if 'relu' in name and not '0.relu' in name:
            module.register_backward_hook(backward_hook_sgm)

        # e.g., 1.layer1.1, 1.layer4.2, ...
        if len(name.split('.')) >= 2 and 'layer' in name.split('.')[-2]:
            module.register_backward_hook(backward_hook_norm)
