"""
This module contains code for segmenting images
"""
import gc
import segmentation_models_pytorch as smp
from pathlib import Path
from magicgui import magicgui, magic_factory
from tqdm import tqdm
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari.layers import Image, Layer, Labels
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari_cool_tools_img_proc import torch,kornia,viewer,device

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)

@magic_factory()
def b_scan_pix2pixHD_seg(img:Image, state_dict_path=Path("D:\JJ\Development\Choroid_Retina_Measurment2\pix2pixHD\checkpoints"), label_flag:bool=True) -> Layer:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name.
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
        Logarithm corrected output image with '_LC' suffix added to name.
    """
    show_info(f'B-scan segmentation thread has started')
    layer = b_scan_pix2pixHD_seg_func(img=img,state_dict_path=state_dict_path,label_flag=label_flag)
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
        Logarithm corrected output image with '_LC' suffix added to name.
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
        torch.cuda.empty_cache()

        memory_stats()    
    
    return layer

@magic_factory()
def enface_unetPlusPlus_seg_func(img:Image, state_dict_path=Path("D:\JJ\Development\Aaron_UNET_Mani_Images-Refactor\out"), label_flag:bool=True) -> Layer:
    """Function runs image/volume through pixwpixHD trained generator network to create segmentation labels. 
    Args:
        img (Image): Image/Volume to be segmented.
        state_dict_path (Path): Path to state dictionary of the network to be used for inference.
        label_flag (bool): If true return labels layer with relevant masks as unique label values
                           If false returns volume with unique channels masked with value 1.
        
    Returns:
        Logarithm corrected output image with '_LC' suffix added to name.
    """
    from jj_nn_framework.image_funcs import normalize_in_range, pad_to_target_2d, pad_to_targetM_2d, bw_1_to_3ch

    pttm_params = {
        'h': 864,
        'w': 864,
        'X_data_format': 'NCHW',
        'y_data_format': 'NHW',
        'mode': 'constant',
        'value': None,
        'device': DEVICE
    }

    data = img.data.copy()
    pt_data = torch.tensor(data,device=device)
    ch3_data = bw_1_to_3ch(pt_data,data_format='NHW')
    pad_data = pad_to_targetM_2d(ch3_data,)

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}
    layer_type = "image"

    ENCODER = "efficientnet-b6"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = [
        "vessel"
    ]
    ACTIVATION = "sigmoid"

    model = smp.UnetPlusPlus(encoder_name=ENCODER, # smp.Unet(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    classes=len(CLASSES),
                    activation=ACTIVATION)
    
    model.eval()
    output = model.predict(pt_data)
    seg_out = output.detach().cpu().numpy()
    layer = Layer.create(seg_out,add_kwargs,layer_type)
    
    print(type(model))
    return layer
    