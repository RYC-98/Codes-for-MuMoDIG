import torch
import torch.nn.functional as F
import torchvision.transforms as T
import kornia.augmentation as K
import random
from functools import partial

from timm.models import create_model

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from ..lb_quantization import LBQuantization

import scipy.stats as st
# import random

class MUMODIG_PNAPO(MIFGSM):


    # 注意测试时是 num_scale=5
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., 
                 N_trans = 6, N_base = 1, N_intepolate = 1, region_num = 2, lamb = 0.65, # 6,1,1,2,0.65
                 targeted=False, random_start=False, 
                 norm='linfty', loss='crossentropy', device=None, attack='EPIG', **kwargs): 

        self.model_name = model_name 

        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack) 
        self.N_trans = N_trans    
        self.N_base = N_base
        self.N_intepolate = N_intepolate    
        self.quant = LBQuantization(region_num)
        self.lamb = lamb


        self._register_model()
        self.image_size = 224
        self.crop_length = 16
        self.max_num_patches = int((224/16)**2)
        self.sample_num_patches = 130


    def load_model(self, model_name): 

        if self.model_name == 'vit_base_patch16_224':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/vit_base_patch16_224.bin')) #                
        elif self.model_name == 'pit_b_224':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/pit_b_224.bin')) #                
        elif self.model_name == 'pit_ti_224':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/pit_ti_224.bin')) #                        
        elif self.model_name == 'cait_s24_224':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/cait_s24_224.bin')) #                
        elif self.model_name == 'visformer_small':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/visformer_small.bin')) #                
        elif self.model_name == 'visformer_tiny':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/visformer_tiny.bin')) #                
        elif self.model_name == 'deit_base_distilled_patch16_224':
            model = create_model(model_name= self.model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/deit_base_distilled_patch16_224.bin')) #                
                        

        model = wrap_model(model.eval().cuda())
        return model
    

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
        delta = self.init_delta(data) # 

        momentum = 0


        for epoch_idx in range(self.epoch):

            delta_patchout = self._generate_samples_for_interactions(delta, epoch_idx) # use epoch_idx as seed

            sole_grad = self.ig(data, delta_patchout, label)
            exp_grad = self.exp_ig(data, delta_patchout, label)
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
        return x.roll(step, dims=2)   
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



    def _register_model(self):
        """
        Register the backward hook for the attention dropout
        (This code is copied from https://github.com/zhipeng-wei/PNA-PatchOut)
        """
        def attn_drop_mask_grad(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            return (mask * grad_in[0][:], )


        drop_hook_func = partial(attn_drop_mask_grad, gamma=0) 

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model[1].blocks[i].attn.attn_drop.register_backward_hook(drop_hook_func)


        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model[1].transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)

        if self.model_name in ['pit_ti_224' ]: # 

            for block_ind in range(12):                   
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind = block_ind            
                elif block_ind < 8 and block_ind >= 2:
                    transformer_ind = 1
                    used_block_ind = block_ind - 2        
                elif block_ind < 12 and block_ind >= 8:
                    transformer_ind = 2
                    used_block_ind = block_ind - 8        
                self.model[1].transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
        

        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model[1].blocks[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind > 24:
                    self.model[1].blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model[1].stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model[1].stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)
        elif self.model_name == 'visformer_tiny':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model[1].stage2[block_ind].attn.attn_drop.register_backward_hook(drop_hook_func)
                elif block_ind >=4:
                    self.model[1].stage3[block_ind-4].attn.attn_drop.register_backward_hook(drop_hook_func)


    def _generate_samples_for_interactions(self, perts, seed):
        """
        Generate masked perturbations w.r.t. the patchout strategy
        (This code is copied from https://github.com/zhipeng-wei/PNA-PatchOut)
        """
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)  

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_patches)]        
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_patches])         

        rows, cols = ids // grid_num_axis, ids % grid_num_axis

        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation
