import torch
import torch.nn.functional as F
from jj_nn_framework.constants import IMAGE_DATA_FORMAT

def bw_1_to_3ch(img_tensor,data_format='NCHW'):
    ''''''
    if data_format == IMAGE_DATA_FORMAT[0]:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)   # N = 1, C = 1, H, W
    elif data_format == IMAGE_DATA_FORMAT[1]:
        img_tensor = img_tensor.unsqueeze(0)                # N = 1, C,H,W
    elif data_format == IMAGE_DATA_FORMAT[2]:
        img_tensor = img_tensor.unsqueeze(1)                # N, C = 1, H,W
    elif data_format == IMAGE_DATA_FORMAT[3]:
        pass                                                # N,C,H,W
    else:
        print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
        assert(True == False)

    try:
        assert(img_tensor.size()[1] == 1)
    except:
        print(f"Input has {img_tensor.size()[1]} channels this function only accepts single channel data!!\n")
        return img_tensor
    else:
        img_3ch = img_tensor.expand(-1,3,-1,-1)          # expand channel dimension to create grayscale color image

    return img_3ch

def normalize_in_range(img: torch.Tensor, min_val:float, max_val:float) -> torch.Tensor:
    ''''''
    norm_tensor = (max_val - min_val) * ((img-img.min())/ (img.max()-img.min())) + min_val
    return norm_tensor

def pad_to_target_2d(img_tensor,target_shape,device='cpu'):
    ''''''
    
    if len(img_tensor.size()) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) # N = 1, C = 1, H, W
    elif len(img_tensor.size()) == 3:

        if img_tensor.size()[0] == 1 or img_tensor.size()[0] == 3: # assumes index 0 is C for single channel or 3 channel RGB image
            img_tensor = img_tensor.unsqueeze(0)
        else:                                                      # assumes index 0 is N aka batch size
            img_tensor = img_tensor.unsqueeze(1)
    elif len(img_tensor.size()) == 4:
        pass
    else:
        print(f"Image shape is not of valid format (N,C,H,W), (N,H,W), or (C,H,W)")
        assert(True == False)
        
    # pad image to appropriate size in this case 800 by 800
    pad_b = img_tensor.size()[0]
    pad_c = img_tensor.size()[1]
    pad_v = int((target_shape[0] - img_tensor.size()[2])/2)
    pad_h = int((target_shape[1] - img_tensor.size()[3])/2)
    
    print(pad_b,pad_v,pad_h)
    
    add_v = torch.zeros((pad_b,pad_c,pad_v,img_tensor.size()[3]),device=device)
    
    img_v = torch.cat((add_v,img_tensor,add_v),dim=2)
    
    print(img_v.size())
    
    add_h = torch.zeros((pad_b,pad_c,img_v.size()[2],pad_h),device=device)
    img_h = torch.cat((add_h,img_v,add_h),dim=3)
    
    return img_h

def pad_to_targetM_2d(img_tensor,target_shape,data_format=None,mode='constant',value=None):
    ''''''

    num_dims = len(img_tensor.size())

    if data_format == None:
        if  num_dims ==  2:
            data_format = 'HW'
        elif num_dims == 3:
            if img_tensor.size()[0] == 1 or img_tensor.size()[0] == 3:
                data_format = 'CHW'
            else:
                data_format = 'NHW'
        elif num_dims == 4:
            data_format = 'NCHW'
        else:
            print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
            assert(True == False)

    
    if data_format == IMAGE_DATA_FORMAT[0]:
        try:
            assert(img_tensor.dim() == 2)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)   # N = 1, C = 1, H, W
    elif data_format == IMAGE_DATA_FORMAT[1]:
        try:
            assert(img_tensor.dim() == 3)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(0)                # N = 1, C,H,W
    elif data_format == IMAGE_DATA_FORMAT[2]:
        try:
            assert(img_tensor.dim() == 3)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            img_tensor = img_tensor.unsqueeze(1)                # N, C = 1, H,W
    elif data_format == IMAGE_DATA_FORMAT[3]:
        try:
            assert(img_tensor.dim() == 4)
        except:
            print(f"Image dimensions {img_tensor.dim()} do not match data format dimensions {data_format}!!\n")
        else:
            pass                                                # N,C,H,W
    else:
        print(f"Image shape is not of valid format (H,w), (C,H,W), (N,H,W), or (N,C,H,W)\n")
        assert(True == False)
        
    # pad image to appropriate size in this case 800 by 800
    pad_b = img_tensor.size()[0]
    pad_c = img_tensor.size()[1]
    pad_v = int((target_shape[0] - img_tensor.size()[2])/2)
    pad_h = int((target_shape[1] - img_tensor.size()[3])/2)
    
    #print(pad_b,pad_v,pad_h)
    #
    #add_v = torch.zeros((pad_b,pad_c,pad_v,img_tensor.size()[3]),device=device)
    #
    #img_v = torch.cat((add_v,img_tensor,add_v),dim=2)
    #
    #print(img_v.size())
    #
    #add_h = torch.zeros((pad_b,pad_c,img_v.size()[2],pad_h),device=device)
    #img_h = torch.cat((add_h,img_v,add_h),dim=3)
    
    img_h = F.pad(img_tensor,(pad_h,pad_h,pad_v,pad_v),mode=mode,value=value)

    return img_h