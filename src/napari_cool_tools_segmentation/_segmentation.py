"""
This module contains code for segmenting images
"""
import gc
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
from typing import List
from magicgui import magic_factory
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer
from napari_cool_tools_io import torch,viewer,device,memory_stats



@magic_factory()
def b_scan_pix2pixHD_seg(img:Image, state_dict_path=Path("D:\JJ\Development\Choroid_Retina_Measurment2\pix2pixHD\checkpoints"), label_flag:bool=True):
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    b_scan_pix2pixHD_seg_thread(img=img,state_dict_path=state_dict_path,label_flag=label_flag)
    return

@thread_worker(connect={"returned": viewer.add_layer},progress=True)
def b_scan_pix2pixHD_seg_thread(img:Image, state_dict_path=Path("D:\JJ\Development\Choroid_Retina_Measurment2\pix2pixHD\checkpoints"), label_flag:bool=True) -> Layer:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    show_info(f'B-scan segmentation thread has started')
    layer = b_scan_pix2pixHD_seg_func(img=img,state_dict_path=state_dict_path,label_flag=label_flag)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'B-scan segmentation thread has completed')
    return layer

def b_scan_pix2pixHD_seg_func(img:Image, state_dict_path=Path("D:\JJ\Development\Choroid_Retina_Measurment2\pix2pixHD\checkpoints"), label_flag:bool=True) -> Layer:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    from models.pix2pixHD_model import InferenceModel
    model = InferenceModel()
    #model.initialize()
    state_dict = torch.load(state_dict_path)
    #model.load_network(model.netG,'G','latest',state_dict_path)
    #model.netG.load_state_dict(state_dict)
    from models.networks import define_G

    def_g_settings = {
        "input_nc": 3,
        "output_nc": 3,
        "ngf": 64,
        "netG": 'global',
        "n_downsample_global": 4,
        "n_blocks_global": 9,
        "n_local_enhancers": 1,
        "n_blocks_local": 3,
        "norm": 'instance',
        "gpu_ids": [0],
    }

    data = img.data.copy()

    try:
        assert data.ndim == 2 or data.ndim == 3, "Only works for data of 2 or 3 dimensions"
    except AssertionError as e:
        print("An error Occured:", str(e))
    else:

        pt_data = torch.tensor(data,device=device)
    
        gen = define_G(**def_g_settings)
        gen.load_state_dict(state_dict)
        gen.eval()
        
        name = f"{img.name}_Seg"
        add_kwargs = {"name":f"{name}"}

        if data.ndim == 2:
            pt_data2 = pt_data.unsqueeze(0).repeat(3,1,1)
            output = gen(pt_data2)
            retina = output[0] == 1
            choroid = output[1] == 1
            sclera = output[2] == 1

            if label_flag:
                labels = torch.zeros_like(output[0])
                labels[retina] = 1
                labels[choroid] = 2
                labels[sclera] = 3
                labels = labels.to(torch.uint8)
                labels_out = labels.detach().cpu().numpy()
                layer_type = 'labels'
                layer = Layer.create(labels_out,add_kwargs,layer_type)

                #clean up
                del labels_out
                del labels
                
            else:
                output2 = output.detach().cpu().numpy()
                layer_type = 'image'                
                layer = Layer.create(output2,add_kwargs,layer_type)
                
                #clean up
                del output2

            #clean up
            del retina
            del choroid
            del sclera
            del output
            del pt_data2
        
        elif data.ndim == 3:
            outstack = []
            for i in tqdm(range(len(data)),desc="Current image"):

                temp_data = pt_data[i].unsqueeze(0).repeat(3,1,1)
                output = gen(temp_data)
                retina = output[0] == 1
                choroid = output[1] == 1
                sclera = output[2] == 1

                if label_flag:
                    labels = torch.zeros_like(output[0])
                    labels[retina] = 1
                    labels[choroid] = 2
                    labels[sclera] = 3

                    outstack.append(labels)
                    #clean up
                    del labels

                else:
                    outstack.append(output)
                    #clean up
                    del output

                #clean up
                del retina
                del choroid
                del sclera
                del output
                del temp_data

                #gc.collect()
                #torch.cuda.empty_cache()

            if label_flag:
                labels2 = torch.stack(outstack)
                labels2 = labels2.to(torch.uint8)
                labels_out = labels2.detach().cpu().numpy()
                layer_type = 'labels'
                layer = Layer.create(labels_out,add_kwargs,layer_type)

                # clean up
                del labels_out
                del labels2
                del outstack
            else:
                output2 = torch.stack(outstack)
                layer_type = 'image'
                layer = Layer.create(output2,add_kwargs,layer_type)

                # clean up
                del output2
                del outstack

        #clean up
        del pt_data
        del gen
        gc.collect()
        #torch.cuda.empty_cache()

        #memory_stats()    
    
    return layer

@magic_factory()
def enface_unet_seg(img:Image, state_dict_path=Path("D:\JJ\Development\Aaron_UNET_Mani_Images-Refactor\out_dict"), label_flag:bool=True):
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Yields:
        Image Layer containing padded enface image with '_Pad' suffix added to name
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    enface_unet_seg_thread(img=img,state_dict_path=state_dict_path,label_flag=label_flag)
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def enface_unet_seg_thread(img:Image, state_dict_path=Path("D:\JJ\Development\Aaron_UNET_Mani_Images-Refactor\out_dict"), label_flag:bool=True) -> List[Layer]:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Yields:
        Image Layer containing padded enface image with '_Pad' suffix added to name
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    show_info(f'Enface segmentation thread has started')
    layers = enface_unet_seg_func(img=img,state_dict_path=state_dict_path,label_flag=label_flag)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f'Enface segmentation thread has completed')
    for layer in layers:
        yield layer
    #return layers

def enface_unet_seg_func(img:Image, state_dict_path=Path("D:\JJ\Development\Aaron_UNET_Mani_Images-Refactor\out_dict"), label_flag:bool=True) -> List[Layer]:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Yields:
        Image Layer containing padded enface image with '_Pad' suffix added to name
        Labels Layer containing B-scan segmentations with '_Seg' suffix added to name.
    """
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch
    from torchvision import transforms
    from kornia.enhance import equalize_clahe

    layers_out = []

    pttm_params = {
        'h': 864,
        'w': 864,
        'X_data_format': 'NCHW',
        'y_data_format': 'NHW',
        'mode': 'constant',
        'value': None,
        'device': device
    }

    data = img.data.copy()
    pt_data = torch.tensor(data,device=device)
    #print(f"pt_data shape: {pt_data.shape}\n")
    ch3_data = bw_1_to_3ch(pt_data,data_format='HW')
    #print(f"ch3_data shape: {ch3_data.shape}\n")
    norm_ch3_data = normalize_in_range(ch3_data,0.0,1.0)
    #print(f"norm_ch3_data shape: {norm_ch3_data.shape}\n")
    pad_data = pad_to_targetM_2d(norm_ch3_data,(864,864),'NCHW')

    name = f"{img.name}_Pad"
    add_kwargs = {"name":f"{name}"}
    layer_type = "image"

    out = pad_data.detach().cpu().numpy().squeeze()

    offset_0 = out[0].shape[0] - data.shape[0]
    offset_1 = out[0].shape[1] - data.shape[1]
    start_0 = int(offset_0/2)
    start_1 = int(offset_1/2)
    end_0 = int(out[0].shape[0] - start_0)
    end_1 = int(out[0].shape[1] - start_1)

    #pad_img = Layer.create(out[0],add_kwargs,layer_type)

    # clean up
    #del pt_data
    #del ch3_data
    #del norm_ch3_data

    #yield pad_img
    #del pad_img
    #layers_out.append(pad_img)

    x = normalize_in_range(pad_data,0,1)
    mean,std = x.mean([0,2,3]),x.std([0,2,3])
    norm = transforms.Normalize(mean,std)
    x_norm = norm(x)
    x_norm2 = normalize_in_range(x_norm,0,1)

    x_eq = equalize_clahe(x_norm2)

    print(f"x shape: {x_eq.shape}\n")

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}
    layer_type = "labels"

    ENCODER = "efficientnet-b5"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = [
        "vessel"
    ]
    ACTIVATION = "sigmoid"

    model = smp.Unet(encoder_name=ENCODER, # smp.UnetPlusPlus(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=len(CLASSES),
                    activation=ACTIVATION)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    model_dev = model.to(device)
    output = model_dev.predict(x_eq)
    
    seg_out = output.detach().cpu().numpy().squeeze().astype(int)
    final_seg = seg_out[start_0:end_0,start_1:end_1]
    layer = Layer.create(final_seg,add_kwargs,layer_type)
    
    # clean up
    del final_seg
    del seg_out
    del output
    del model_dev
    del model
    del x_eq
    del x_norm2
    del x_norm
    del norm
    del mean
    del std
    del x
    del out
    del pad_data
    del norm_ch3_data
    del ch3_data
    del pt_data

    gc.collect()
    #torch.cuda.empty_cache()
    #memory_stats()

    layers_out.append(layer)
    return layers_out