import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import pandas as pd
import timm
import os

img_height, img_width = 224, 224
img_max, img_min = 1., 0

cnn_model_paper = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224',
                   'visformer_small', 'swin_tiny_patch4_window7_224']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']


def load_pretrained_model(cnn_model=[], vit_model=[]): 
    
    for model_name in cnn_model:
        yield model_name, models.__dict__[model_name](weights="DEFAULT")           
        

    for model_name in vit_model:
        if model_name == 'vit_base_patch16_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/vit_base_patch16_224.bin')) # 
        elif model_name == 'deit_tiny_patch16_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/deit_tiny_patch16_224.bin'))
        elif model_name == 'deit_small_patch16_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/deit_small_patch16_224.bin'))
        elif model_name == 'pit_ti_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/pit_ti_224.bin'))         
        elif model_name == 'pit_b_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/pit_b_224.bin')) # 
        elif model_name == 'visformer_tiny':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/visformer_tiny.bin')) 
        elif model_name == 'visformer_small':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/visformer_small.bin')) #              
        elif model_name == 'swin_tiny_patch4_window7_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/swin_tiny_patch4_window7_224.bin'))
        elif model_name == 'swin_small_patch4_window7_224':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/swin_small_patch4_window7_224.bin'))
        elif model_name == 'coat_tiny':
            yield model_name, timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file='./vit_weight/coat_tiny.bin')) #              
        
        else:
            print(model_name, "in vit_model_paper have not been modified in load_pretrained_model of utils.py")


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    if hasattr(model, 'default_cfg'):      
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
    else:
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    return torch.nn.Sequential(normalize, model)


def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0,2,3,1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))

def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):                          
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)      
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError



class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False): 
        self.targeted = targeted
        self.data_dir = input_dir

        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv')) # default

        if eval:                               
            self.data_dir = output_dir         
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))

        else:                                  
            self.data_dir = os.path.join(self.data_dir, 'images')          
            print('=> Train mode: training on {}'.format(self.data_dir))   
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]  

        assert isinstance(filename, str)       

        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32)/255        
        image = torch.from_numpy(image).permute(2, 0, 1)      
        label = self.f2l[filename]

        return image, label, filename          

    def load_labels(self, file_name):          
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'],
                                             dev.iloc[i]['target_label']] for i in range(len(dev))} 

        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l                             


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
