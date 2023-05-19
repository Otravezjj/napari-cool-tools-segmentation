"""
This module is contains code for segmenting images
"""
from pathlib import Path
from magicgui import magicgui, magic_factory
from napari.layers import Image, Layer, Labels
from napari.types import ImageData, LabelsData, LayerDataTuple
from napari_cool_tools_img_proc import torch,kornia,viewer,device

@magic_factory()
def segmentation_inference(img:Image, state_dict_path=Path("D:\JJ\Development\Choroid_Retina_Measurment2\pix2pixHD\checkpoints"), label_flag:bool=True) -> Layer:
    """
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
    pt_data = torch.tensor(data,device=device)
    

    gen = define_G(**def_g_settings)
    gen.load_state_dict(state_dict)
    gen.eval()
    print(type(state_dict),type(model),model.name(),type(gen))

    print(pt_data.shape)
    pt_data = pt_data.unsqueeze(0).repeat(3,1,1)
    print(pt_data.shape)
    print(pt_data.dtype)
    #pt_data = pt_data.to(torch.uint8)
    #print(pt_data.dtype)
    #print(pt_data.min(),pt_data.max())

    output = gen(pt_data)

    retina = output[0] == 1
    choroid = output[1] == 1
    sclera = output[2] == 1

    print(retina)
    labels = torch.empty_like(output[0])
    labels[retina] = 1
    labels[choroid] = 2
    labels[sclera] = 3
    labels = labels.to(torch.uint8)
    labels = labels.detach().cpu().numpy()

    output = output.detach().cpu().numpy()
    print(output.shape)
    print(output.min(),output.max())

    name = f"{img.name}_Seg"
    add_kwargs = {"name":f"{name}"}

    if label_flag:
        layer_type = 'labels'
        layer = Layer.create(labels,add_kwargs,layer_type)
    else:
        layer_type = 'image'
        layer = Layer.create(output,add_kwargs,layer_type)

    return layer
