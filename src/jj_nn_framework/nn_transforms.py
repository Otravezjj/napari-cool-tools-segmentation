
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import kornia.augmentation as K
from kornia.constants import Resample, SamplePadding
from torchvision import transforms
from kornia.enhance import equalize_clahe
from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
from jj_nn_framework.mod_utils import check_rand_sample_per_crop, get_crops

# custom transforms

class AaronUnetTrainAug(nn.Module):
    def __init__(self,
        hfp,vfp,bp,cp,degrees=90,scale=(0.9,1.1),bf=(0.8,1.2),cf=(0.8,1.2),crs=(224,224),
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        self.RCr = K.RandomCrop(size=crs)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x,y = self.CLAHE((x,y))

        x = self.RBr(x)
        x = self.RCo(x)

        x = self.RCr(x)
        y = self.RCr(y,params=self.RCr._params)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)
            
        return x,y

class JohnUnetTrainAug(nn.Module):
    def __init__(self,
        hfp,vfp,bp,cp,degrees=90,scale=(0.9,1.1),bf=(0.8,1.2),cf=(0.8,1.2),crs=(224,224),cr_shuff=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x,y = self.CLAHE((x,y))

        x = self.RBr(x)
        x = self.RCo(x)

        x,y = self.RCr((x,y))
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)
            
        return x,y

class JohnUnetTrainAug2(nn.Module):
    def __init__(self,
        target_imshape,hfp,vfp,bp,cp,
        X_data_format = 'None',
        y_data_format = 'None',
        mode='constant',value=None,
        bf=(0.8,1.2),
        cf=(0.8,1.2),
        crs=(224,224),cr_shuff=True,
        degrees=90,scale=(0.9,1.1),
        bt_flag=True,
        device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.device=device
        self.PAD = PadToTargetM(h=target_imshape[0],w=target_imshape[1],X_data_format=X_data_format,y_data_format=y_data_format,mode=mode,value=value)
        self.RHF = K.RandomHorizontalFlip(hfp,same_on_batch=False)
        self.RVF = K.RandomVerticalFlip(vfp,same_on_batch=False)
        self.CLAHE = NormalizeCLAHE()
        self.RBr = K.RandomBrightness(brightness=bf,p=bp,same_on_batch=False)
        self.RCo = K.RandomContrast(contrast=cf,p=cp,same_on_batch=False)
        #self.RCr = K.RandomCrop(size=crs)
        self.RCr = ImageCrops(h=crs[0],w=crs[1],shuffle=cr_shuff,device=device,verbose=verbose,sanity=sanity)
        self.RAT = K.RandomAffine(degrees,translate=None,scale=scale,shear=None,resample=Resample.BILINEAR.name,
            padding_mode=SamplePadding.REFLECTION.name,same_on_batch=False,p=1.0
        )
        self.BT = BinaryTarget()
        self.BT_flag = bt_flag
        
    def forward(self, t):
        
        x = t[0]
        y = t[1]

        #print(f"Initial input:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.PAD((x,y))
        #print(f"Post padding:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.CLAHE((x,y))
        #print(f"Post CLAHE:\nx shape: {x.shape}, y shape: {y.shape}\n")
        x,y = self.RCr((x,y))
        
        x = self.RHF(x)
        y = self.RHF(y,params=self.RHF._params)

        x = self.RVF(x)
        y = self.RVF(y,params=self.RVF._params)

        x = self.RBr(x)
        x = self.RCo(x)
                
        x = self.RAT(x)
        y = self.RAT(y,params=self.RAT._params)

        if self.BT_flag:
            x,y = self.BT((x,y))
            
        return x,y

class ImageCrops(nn.Module):
    def __init__(self,h,w,shuffle=True,device='cpu',verbose=False,sanity=False):
        super().__init__()
        
        self.crop_dim = (h,w)
        self.shuffle = shuffle
        self.device = device
        self.verbose = verbose
        self.sanity = sanity

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        print(f"ImageCrops x/y shapes: {x.shape}/{y.shape}\n")

        h,w = x.shape[2],y.shape[3]
        kh,kw = self.crop_dim[0], self.crop_dim[1]

        #print(
        #    f"h: {h}, w: {w}, kh: {kh}, kw: {kw}\n"
        #)
        
        h_div = h/kh
        w_div = w/kw
        h_diff = h - kh
        w_diff = w - kw
        kh_h_div_diff = int((h - (kh*int(h_div))) / 2)
        kw_w_div_diff = int((w - (kw*int(w_div))) / 2)
        
        h_start,h_end = 0, h_diff
        w_start,w_end = 0, w_diff
        
        mid_h_start,mid_h_end = kh_h_div_diff,(h - kh_h_div_diff) - kh
        mid_w_start,mid_w_end = kw_w_div_diff,(w - kw_w_div_diff) - kw
        
        if mid_h_start == 0 and mid_h_end == h - kh:
            mid_h_start,mid_h_end = kh,h - kh
        else:
            pass
        
        if mid_w_start == 0 and mid_w_end == w - kw:
            mid_w_start,mid_w_end = kw,w - kw
        else:
            pass

        #print(
        #    f"mid_h_start: {mid_h_start}\n"
        #    f"mid_h_end: {mid_h_end}\n"
        #    f"kh/2: {int(kh/2)}"
        #)

        mid_h_indicies = torch.arange(mid_h_start,mid_h_end+1,int(kh/2),device=self.device)
        mid_w_indicies = torch.arange(mid_w_start,mid_w_end+1,int(kw/2),device=self.device)
        
        start_h = torch.tensor((h_start,),device=self.device)
        end_h = torch.tensor((h_end,),device=self.device) 
        h_indicies = torch.cat((start_h,mid_h_indicies,end_h)).unique()
        
        start_w = torch.tensor((w_start,),device=self.device)
        end_w = torch.tensor((w_end,),device=self.device) 
        w_indicies = torch.cat((start_w,mid_w_indicies,end_w)).unique()
        
        if self.verbose:
            print(f"h_indicies:\n{h_indicies}\nw_indicies:\n{w_indicies}\n")
        
        crops_x,crops_y = get_crops(x,y,kh,kw,h_indicies,w_indicies,shuffle=self.shuffle,device=self.device,verbose=self.verbose)
        
        if self.verbose:
            print(f"num_crops = {len(crops_x)}, shapes per crop x,y: {crops_x[0].shape},{crops_y[0].shape}\n")
        
        if self.sanity:
            check_rand_sample_per_crop(crops_x,crops_y,h_indicies,w_indicies,device=self.device,verbose=self.verbose)
        
        return (crops_x,crops_y)

class NormalizeCLAHE(nn.Module):
    def __init__(self):
        super().__init__()
        #self.norm_rang = normalize_in_range
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)
        mean,std = x.mean([0,2,3]),x.std([0,2,3])
        #print(x.shape,y.shape)
        #print(mean,std)
        norm = transforms.Normalize(mean,std)
        x = norm(x)
        #x = (x-x.min())/(x.max()-x.min())
        x = normalize_in_range(x,0,1)

        x = equalize_clahe(x)
            
        return (x,y)

class PadToTarget(nn.Module):
    def __init__(self,h,w,device='cpu'):
        super().__init__()
        self.h = h
        self.w = w
        self.device = device
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        x = pad_to_target_2d(x,(self.h,self.w),device=self.device)
        y = pad_to_target_2d(y,(self.h,self.w),device=self.device)
            
        return (x,y) 

class PadToTargetM(nn.Module):
    def __init__(self,h,w,X_data_format=None,y_data_format=None,mode='constant',value=None,device='cpu'):
        super().__init__()
        self.h = h
        self.w = w
        self.X_data_format=X_data_format
        self.y_data_format=y_data_format
        self.mode = mode
        self.value = value
       
    def forward(self, t):
        x = t[0]
        y = t[1]

        x = normalize_in_range(x,0,1)
        y = normalize_in_range(y,0,1)
        
        x = pad_to_targetM_2d(x,(self.h,self.w),data_format=self.X_data_format,mode=self.mode,value=self.value)
        y = pad_to_targetM_2d(y,(self.h,self.w),data_format=self.y_data_format,mode=self.mode,value=self.value)
            
        return (x,y) 

class BinaryTarget(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        y = torch.where(y>0,1,0).to(torch.float32)
            
        return (x,y)

class BWOneToThreeCh(nn.Module):
    def __init__(self,
        data_format='NCHW',
        include_label=False
    ):
        super().__init__()

        self.data_format = data_format
        self.include_label = include_label
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        x = bw_1_to_3ch(x,self.data_format)

        if self.include_label:
            y = bw_1_to_3ch(y,self.data_format)
            
        return (x,y) 

class RcsToTensor(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        
        self.device = device
       
    def forward(self, x):
        x.image = TF.to_tensor(x.image)  #.to(self.device)
        x.label = torch.as_tensor(x.label,dtype=torch.int64).permute(2,0,1)       #device=self.device).permute(2,0,1)
            
        return x 

class RcsChannelFirst(nn.Module):
    def __init__(self,device='cpu'):
        super().__init__()
        
        self.device = device

    def forward(self, t):
        x = t[0]
        y = t[1]

        #remove alpha channel from mask
        y = y[:,:,:,:-1]

        x = x.permute(0,3,1,2)
        y = y.permute(0,3,1,2)
        #x.image = TF.rgb_to_grayscale(x.image)
        #x.label = TF.rgb_to_grayscale(x.label)
            
        return x,y

class RcsToGrayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t):
        x = t[0]
        y = t[1]
        x = TF.rgb_to_grayscale(x)
        y = TF.rgb_to_grayscale(y)
        #x.image = TF.rgb_to_grayscale(x.image)
        #x.label = TF.rgb_to_grayscale(x.label)
            
        return x,y

class RcsResize(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        
        self.new_dim = (h,w)
       
    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = TF.resize(x,self.new_dim)
        y = TF.resize(y,self.new_dim)
        #x.image = TF.resize(x.image,self.new_dim)
        #x.label = TF.resize(x.label,self.new_dim)
            
        return (x,y) 

class RcsRandCrop(nn.Module):
    def __init__(self,h,w):
        super().__init__()
        
        self.crop_dim = (h,w)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        #print(x.shape,y.shape)
        #print(self.crop_dim)
        
        v_diff = x.shape[2] - self.crop_dim[0]
        h_diff = y.shape[3] - self.crop_dim[1]
        #print(v_diff,h_diff)
        
        v = torch.randint(0,v_diff,(1,)).item()
        h = torch.randint(0,h_diff,(1,)).item()
        
        x = TF.crop(x,v,h,self.crop_dim[0],self.crop_dim[1])
        y = TF.crop(y,v,h,self.crop_dim[0],self.crop_dim[1])
        
        return (x,y)

class RcsRandStitchCrop(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)
        
        #print(ph,pw)
        
        x = x[:,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        y = y[:,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        return (x,y)

class RcsFullStitchCrop(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)

        perm = torch.randperm(h_idx*w_idx,device=self.device)
        perm2 = perm.view(h_idx,w_idx)

        #print(perm.shape,perm2.shape)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)

        out_x = torch.empty(len(perm),1,x.shape[1],kh,kw,device=self.device)
        out_y = torch.empty(len(perm),1,y.shape[1],kh,kw,device=self.device)
        
        for i in range(len(perm)):
            #print((perm2==i).nonzero().squeeze())
            ph,pw = (perm2 == i).nonzero().squeeze()
            out_x[i] = x[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
            out_y[i] = y[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        #print(ph,pw)
        
        #x = x[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        #y = y[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        
        return (out_x.squeeze(1),out_y.squeeze(1))

class RcsFullStitchCrop2(nn.Module):
    def __init__(self,kh,kw,device):
        super().__init__()
        
        self.kh = kh
        self.kw = kw
        self.device = device

    def forward(self, t):
        # expecting N,C,H,W format
        x = t[0]
        y = t[1]

        kh,kw = self.kh,self.kw
        sh,sw = int(self.kh/2),int(self.kw/2)
        
        h_idx = int(x.shape[2] / self.kh)
        w_idx = int(x.shape[3] / self.kw)

        perm = torch.randperm(h_idx*w_idx,device=self.device)
        perm2 = perm.view(h_idx,w_idx)

        #print(perm.shape,perm2.shape)
        
        #print(h_idx,w_idx)
        
        ph = torch.randint(0,h_idx,(1,),device=self.device)
        pw = torch.randint(0,w_idx,(1,),device=self.device)

        out_x = torch.empty(len(perm),1,x.shape[1],kh,kw,device=self.device)
        out_y = torch.empty(len(perm),1,y.shape[1],kh,kw,device=self.device)
        
        for i in range(len(perm)):
            #print((perm2==i).nonzero().squeeze())
            ph,pw = (perm2 == i).nonzero().squeeze()

            #if i%2 == 0:
            out_x[i] = x[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
            out_y[i] = y[0,:,ph*kh:ph*kh+kh,pw*kw:pw*kw+kw]
        
        #print(ph,pw)
        
        #x = x[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        #y = y[:,:,ph*40:ph*40+40,pw*40:pw*40+40]
        
        return (out_x.squeeze(1),out_y.squeeze(1))

class RcsNormalize(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        
        x = (x-x.min())/(x.max()-x.min())
        mean,std = x.mean([0,2,3]),x.std([0,2,3])
        #print(x.shape,y.shape)
        #print(mean,std)
        norm = transforms.Normalize(mean,std)
        x = norm(x)
        x = (x-x.min())/(x.max()-x.min())
            
        return (x,y) 

class RcsBinaryTarget(nn.Module):
    def __init__(self):
        super().__init__()
       
    def forward(self, t):
        x = t[0]
        y = t[1]
        y = torch.where(y>0,1,0).to(torch.float32)
            
        return (x,y) 


class RcsFlipsRot(nn.Module):
    def __init__(self,
        hfp,vfp,degrees,
        device='cpu'):
        super().__init__()
        
        self.device=device
        self.RHF = transforms.RandomHorizontalFlip(p=hfp)
        self.RVF = transforms.RandomVerticalFlip(p=vfp)
        self.Rot = transforms.RandomRotation(degrees=degrees)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RHF(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RHF(y)
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RVF(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RVF(y)
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.Rot(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.Rot(y)
        
            
        return x,y


class RcsNonRotAff(nn.Module):
    def __init__(self,
        translate,scale,shear,
        device='cpu'):
        super().__init__()
        
        self.RA = transforms.RandomAffine(degrees=0,translate=translate,scale=scale,shear=shear)
        self.device=device

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        seed = torch.randint(0,255,(1,),device=self.device)
        torch.manual_seed(seed)
        x = self.RA(x)
        torch.random.fork_rng()
        torch.manual_seed(seed)
        prev_state = torch.get_rng_state()
        torch.random.set_rng_state(prev_state)
        y = self.RA(y)        
            
        return x,y

class RcsGBlur(nn.Module):
    def __init__(self,
        kernel_size,sigma=(0.1,0.2),
        device='cpu'):
        super().__init__()
        
        self.Gblur = transforms.GaussianBlur(kernel_size,sigma=sigma)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.Gblur(x)
           
        return x,y

class RcsSharpCon(nn.Module):
    def __init__(self,
        sf,sp=0.5,acp=0.5,
        device='cpu'):
        super().__init__()
        
        self.RS = transforms.RandomAdjustSharpness(sf,sp)
        self.AC = transforms.RandomAutocontrast(p=acp)

    def forward(self, t):
        
        x = t[0]
        y = t[1]
        
        x = self.AC(x)
        x = self.RS(x)
           
        return x,y