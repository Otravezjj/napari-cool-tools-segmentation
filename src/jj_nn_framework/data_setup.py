"""
Functions for loading data .
"""
import gc
import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
from skimage import io
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from jj_nn_framework.utils import get_config

class TrainingSplits():
    ''''''
    def __init__(self,X_train,y_train,X_test,y_test,X_val=None,y_val=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        if X_val != None and y_val != None:
            self.X_val = X_val
            self.y_val = y_val
    
    def __str__(self):
        ''''''
        out_str = "Dataset containing:\n"
        for i,key in enumerate(vars(self).keys()):
            if (i-1)%2 == 0:
                x = f"{key}: {vars(self)[key].shape}\n"
            else:
                x = f"{key}: {vars(self)[key].shape}, "
            out_str = out_str + x
        return out_str
            

def load_mnist_img_label_splits(data_dir, val_split=0.0):
    ''''''
    
    dataset_tuple = load_mnist_datasets(data_dir)
    
    training = dataset_tuple[0]
    testing = dataset_tuple[1]
    
    X_train,y_train = training.data,training.targets
    X_test,y_test = testing.data,testing.targets
    
    if val_split == 0:
        return TrainingSplits(X_train,y_train,X_test,y_test)
    elif val_split >= 1.0 or val_split < 0:
        print("val_split must be >= 0 and < 1 as it indicates the percentage split\
 between testing and validation sets\n")
    else:
        samples = len(X_test)
        val_samples = int(val_split * samples)
        test_samples = samples - val_samples
        X_val,y_val = X_test[test_samples:],y_test[test_samples:]
        X_test,y_test = X_test[:test_samples],y_test[:test_samples]
        
        return TrainingSplits(X_train,y_train,X_test,y_test,X_val,y_val)
    

def load_mnist_datasets(data_dir):
    ''''''
    
    data_path = Path(os.path.abspath(data_dir))
    try:
        assert data_path.is_dir()
    except:
        print(f"\nNo valid data_path directory @:\n{data_path}\n")
    else:
        #return data_path
        
        # Datasets
        MNIST_training = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        MNIST_testing = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
        
        return MNIST_training,MNIST_testing
    
def create_dataloaders(
    config_dir: str,
    data_dir: str,
    val_split = 0,
):
    ''''''
    
    # get configuration
    config_data = get_config(config_dir)
    dl_config = config_data["dataloader_params"]
    batch_sz = dl_config["batch_sz"]
    num_workers = 0 #dl_config["num_workers"]
    shuffle = bool(dl_config["shuffle"])
    
    # define data loader parameters
    # parameters
    params = {
        'batch_size':batch_sz,
        'shuffle':shuffle,
        'num_workers':num_workers,
        'pin_memory':True
    }
    
    # get dataset splits
    splits = load_mnist_img_label_splits(data_dir,val_split)
    
    datasets = datasets_from_splits(splits)
    
    
    dataloaders = {}
    
    for key,dataset in datasets.items():
        print(f"key:{key}\ndataset:\n{dataset}\n")
        dataloaders[key] = DataLoader(dataset,**params)
    
            
    return dataloaders

def datasets_from_splits(splits):
    ''''''
    datasets = {}
    # preprocess image data
    for key in vars(splits).keys():
        if "X" in key:
            y_key = key.replace("X","y")
            X_raw = vars(splits)[key]
            y_raw = vars(splits)[y_key]
            #vars(splits)[key] = (vars(splits)[key]/256).to(torch.float32)
            vars(splits)[key],vars(splits)[y_key] = preproc_uint8_norm_1hot(X_raw,y_raw)
            X_out,y_out = vars(splits)[key],vars(splits)[y_key]
            
            if "train" in key:
                o_key = "training"
            elif "test" in key:
                o_key = "testing"
            elif "val" in key:
                o_key = "validation"
            else:
                o_key = "misc"
            
            datasets[o_key] = DatasetImgLabel(X_out,y_out)
            
            
    return datasets
    

class DatasetImgLabel(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_data,labels):
        '''Initialization'''
        
        try:
            assert(img_data != None and labels != None)
        except:
            print(f"Image or Labels data is missing !!")
        else:
            self.img_data = img_data
            self.labels = labels
        
        
    def __len__(self):
        '''Returns total number of samples'''
        return len(self.labels)
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''        
        
        # get data and label        
        return self.img_data[idx].unsqueeze(0),self.labels[idx]
    
    def __str__(self):
        return f"Image, Label Dataset: {type(self.img_data).__name__}{list(self.img_data.shape)}({self.img_data.dtype}), \
{type(self.labels).__name__}{list(self.labels.shape)}({self.labels.dtype})"
    
    def __repr__(self):
        return f"DatasetImgLabel(img_data={type(self.img_data).__name__},labels={type(self.labels).__name__})"
    
def preproc_uint8_norm_1hot(X_vec,y_scalar):
    ''''''
    
    X_out = X_vec/256
    num_class = y_scalar.max() + 1
    num_samples = y_scalar.shape[0]
    y_out = torch.zeros(num_samples,num_class,dtype=torch.uint8)
    y_range = torch.arange(0,num_samples,dtype=torch.int64)
    y_out[y_range,y_scalar[y_range]] = 1
    
    return X_out,y_out


def gen_files(img_paths,label_paths,num_samples,data_path,file_name,file_dir):
        ''''''
        names = []
        images = []
        labels = []

        for i in range(num_samples):

            img_name = img_paths[i].as_posix()
            lbl_name = label_paths[i].as_posix()
            img_type = img_paths[i].suffixes
            lbl_type = label_paths[i].suffixes
            names.append({'img_name':img_name, 'lbl_name':lbl_name})

            try:
                assert(len(img_type) == 1 and len(lbl_type) == 1)
            except:
                print(f"Cannot open multipart file extension/s {img_type if len(img_type)>1 else ''} and/or {lbl_type if len(lbl_type)>1 else ''} are not supported\n")
            else:
                img_type = img_type[0]
                lbl_type = lbl_type[0]

            if img_type == '.npy':
                print("inside im_type == '.npy'")
                img = np.load(img_paths[i])
            elif img_type == '.png':
                img = io.imread(img_paths[i])
            else:
                print(f"File type {img_type} is not supported\n")

            if lbl_type == '.npy':
                label = np.load(label_paths[i])
            elif lbl_type == '.png':
                label = io.imread(label_paths[i])
            else:
                print(f"File type {img_type} is not supported\n")
            
            images.append(img)
            labels.append(label)

        images = torch.from_numpy(np.array(images))
        labels = torch.from_numpy(np.array(labels))

        save_path = data_path/file_dir

        os.makedirs(save_path,exist_ok=True)

        torch.save(
            {
                'file_paths':names,
                'images':images,
                'masks':labels
            }, save_path / f"{file_name}.pt"
        )
        
        print(f"{file_name} was saved to:\n{save_path}\nContains indexed file paths, images and labels.\n")

        return names,images,labels
        
        ''''''
        img_path = Path(img_dir)
        label_path = Path(label_dir)
        data_path = img_path.parent


def gen_pt_files(img_dir,label_dir,file_name:str='image_label_data',file_dir = 'unified_Pytorch_data'):
    ''''''
    img_path = Path(img_dir)
    label_path = Path(label_dir)
    data_path = img_path.parent
        
    try:
        assert(img_path.is_dir() and label_path.is_dir())
    except:
        print(f"Image or Labels data is missing !!")
    else:
        imgs = [x for x in img_path.iterdir() if x.is_file()]
        labels = [y for y in label_path.iterdir() if y.is_file()]
        num_imgs = len(imgs)
        num_labels = len(labels)
        try:
            assert(num_imgs == num_labels)
        except:
            print(f"Number of images and labels do not match !!\n"
                f"There are {num_imgs} images and {num_labels} labels.\n")
        
        else:
            out = gen_files(imgs,labels,num_imgs,data_path,file_name=file_name,file_dir=file_dir)
            return out

def load_img_lbl_tensors(data_dir,device='cpu'):
    ''''''
    
    data_path = Path(data_dir)

    try:
        assert(data_path.is_file())
        assert(data_path.suffix == '.pt')
    except:
        print(f"Image or Labels data are missing !!")
    else:
        
        data = torch.load(data_path)
        if "file_paths" in data.keys():
            paths = data["file_paths"]
        else:
            pass
        
        images = data["images"]
        labels = data["masks"] # change this to conditionally accept "masks" or "labels"

        num_imgs = len(images)
        num_labels = len(labels)

        try:
            assert(num_imgs == num_labels)
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {num_imgs} images and {num_labels} labels.\n")
        else:
            return(images.to(device),labels.to(device))


class RetCamTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, images,labels,transform=None,device='cpu'):
        '''Initialization'''
        
        try:
            assert(len(images) == len(labels))
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {len(images)} images and {len(labels)} labels.\n")
            return(images,labels)
        
        else:
            self.images = images
            self.labels = labels
            self.num_samples = len(images)
            self.transform = transform
            self.device = device
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        image = self.images[idx] / 256 #### comment something here
        label = self.labels[idx]
            
        if self.transform:
            data = self.transform((image,label))
            image = data[0]
            label = data[1]
        else:
            return image,label
    
    def __str__(self):
        str_out = (f"RetCamDataset:\nContains {self.num_samples} images with labels\nimages:\n{self.images.shape}\n"
                   f"labels:\n{self.labels.shape}\n")
        return str_out
    
    def __repr__(self):
        return f"RetCamDataset(images={torch.Tensor.__name__},labels={torch.Tensor.__name__})"
    
    def generate_split_datasets(self,per_tr=.8,per_t=.1,per_v=.1,possible_splits=20):
        ''''''
        num_train = round(self.num_samples*per_tr)
        num_test = round(self.num_samples*per_t)
        num_val = round(self.num_samples*per_v)
        
        print(f"train:{num_train}, test:{num_test}, val:{num_val} = {self.num_samples}?"
              f"{(num_train+num_test+num_val)==self.num_samples}")
        
        manual_seed = torch.randint(possible_splits,(1,)).item()
        torch.manual_seed(manual_seed)
        idxs = torch.randperm(self.num_samples)
        
        train_idx = idxs[:num_train]
        test_idx = idxs[num_train:num_train+num_test]
        val_idx = idxs[num_train+num_test:]
        
        print(f"train_idx:{train_idx.shape},test_idx{test_idx.shape},val_idx{val_idx.shape}\n")
        
        train_imgs = self.images[train_idx]
        train_lbls = self.labels[train_idx]
        
        train_dset = RetCamTensorDataset(train_imgs,train_lbls,transform=self.transform,device=self.device)
        train_dl = DataLoader(train_dset)
        
        training = {
            'type':'training',
            'manual_seed':manual_seed,
            'dataset':train_dset
        }
        
        test_imgs = self.images[test_idx]
        test_lbls = self.labels[test_idx]
        
        test_dset = RetCamTensorDataset(test_imgs,test_lbls,transform=self.transform,device=self.device)
        test_dl = DataLoader(test_dset)
        
        testing = {
            'type':'testing',
            'manual_seed':manual_seed,
            'dataset':test_dset
        }
        
        val_imgs = self.images[val_idx]
        val_lbls = self.labels[val_idx]
        
        val_dset = RetCamTensorDataset(val_imgs,val_lbls,transform=self.transform,device=self.device)
        val_dl = DataLoader(val_dset)
        
        validation = {
            'type':'validation',
            'manual_seed':manual_seed,
            'dataset':val_dset
        }
        
        return {'train':training,'test':testing,'val':validation}
    
    def check_samples(self):
        ''''''
        
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = self[idx]
            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[1]}x{image.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(image):
                    plt.imshow(image.detach().to('cpu'))
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[1]}x{label.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(label):
                    plt.imshow(label.detach().to('cpu'))
                else:
                    pass

        plt.show()


class BasicTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, images,labels,transform=None,device='cpu'):
        '''Initialization'''
        
        try:
            assert(len(images) == len(labels))
        except:
            print(f"Number of images and labels do not match !!\n"
                  f"There are {len(images)} images and {len(labels)} labels.\n")
            return(images,labels)
        
        else:
            self.images = images
            self.labels = labels
            self.num_samples = len(images)
            self.transform = transform
            self.device = device
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples
    
    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        image = self.images[idx] 
        label = self.labels[idx]
            
        if self.transform:
            data = self.transform((image,label))
            image = data[0]
            label = data[1]
        else:
            return image,label

    def check_samples(self):
        ''''''
        
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = self[idx]
            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[1]}x{image.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(image):
                    plt.imshow(image.detach().permute(1,2,0).to('cpu'))
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[1]}x{label.shape[2]}')
                plt.axis('off')
                if torch.is_tensor(label):
                    plt.imshow(label.detach().permute(1,2,0).to('cpu'))
                else:
                    pass

        plt.show()

class LoadTensorDataset(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor_dir,transform=None,preprocessing=None,device='cpu'):
        '''Initialization'''

        img_lbl_tensor_path = Path(img_lbl_tensor_dir)
        
        try:
            assert(img_lbl_tensor_path.is_dir())
        except:
            print(f"{img_lbl_tensor_path} is not a valid path!!\n")
        
        else:
            batches = list(img_lbl_tensor_path.glob('*.pt'))

            image,label = load_img_lbl_tensors(batches[0],device='cpu')

            self.num_samples = len(image)
            self.batches = batches
            self.num_batches = len(batches)
            self.transform = transform
            self.preprocessing = preprocessing
            self.device = device
            del image
            del label
            #images,labels = load_img_lbl_tensors(img_lbl_tensor_path,device=device)
            #self.images = images
            #self.labels = labels
            #self.num_samples = len(images)
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_batches

    def __getitem__(self, idx):
        '''Generates a single sample of the data'''
        
        img_lbl_tensor_f = self.batches[idx]
        img_lbl_tensor_path = Path(img_lbl_tensor_f)
        image,label = load_img_lbl_tensors(img_lbl_tensor_path,device=self.device)
            
        if self.transform:
            print("Transforming data.\n")
            image,label = self.transform((image,label))
            #print(image.shape)
            self.num_samples = len(image)
        else:
            pass

        if self.preprocessing:
            print("Preprocessing data.\n")
            image,label = self.preprocessing((image,label))    
            
        return image,label

    def check_samples(self):
        ''''''

        image_batch,label_batch = self[0]
        #print(image_batch.shape,label_batch.shape)

        print(f"num_samples: {self.num_samples}\n")
        if self.num_samples < 8:
            idxs = torch.tensor([0])
        else:
            #idxs = torch.randint(0,self.num_samples,(16,))
            perm = torch.randperm(self.num_samples)
            idxs = perm[:16] 
            #idxs = torch.tensor((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),dtype=torch.uint8) # comment out to return to random

        fig = plt.figure(figsize=(16,12))
        fig.suptitle('Sanity Check for Image/Label Data')

        for i,idx in enumerate(idxs):
            image,label = image_batch[idx],label_batch[idx]

            if image.dim() == 2:
                im_h_dim,im_w_dim,im_ch = 0,1,1
            elif image.dim() == 3:
                im_h_dim,im_w_dim,im_ch = 1,2,image.size()[0]
            else:
                pass # error checking here or elsewhere

            if label.dim() == 2:
                lb_h_dim,lb_w_dim,lb_ch = 0,1,1
            elif label.dim() == 3:
                lb_h_dim,lb_w_dim,lb_ch = 1,2,label.size()[0]
            else:
                pass # error checking here or elsewhere

            #print(sample,i)
            if i % 2 == 0:
                fig.add_subplot(4, 4, i+1)
                plt.tight_layout()
                plt.title(f'Sample {i} @\n idx: {idx}\n{image.shape[im_h_dim]}x{image.shape[im_w_dim]}x{im_ch}')
                plt.axis('off')
                if torch.is_tensor(image):
                    if len(image.size()) == 3:
                        plt.imshow(image.detach().to('cpu').permute(1,2,0))
                    elif len(image.size()) == 2:
                        plt.imshow(image.detach().to('cpu'))
                    else:
                        pass # probably need error checking here or earlier
                else:
                    pass

                fig.add_subplot(4, 4, i+2)
                plt.tight_layout()
                plt.title(f'Label {i} @\n idx: {idx}\n{label.shape[lb_h_dim]}x{label.shape[lb_w_dim]}x{lb_ch}')
                plt.axis('off')
                if torch.is_tensor(label):
                    if len(label.size()) == 3:
                        plt.imshow(label.detach().to('cpu').permute(1,2,0))
                    elif len(label.size()) == 2:
                        plt.imshow(label.detach().to('cpu'))
                    else:
                        pass # probably need error checking here or earlier
                else:
                    pass

        plt.show()


class LoadTensorDataset2(Dataset):
    '''Characterizes a dataset for Pytorch'''
    def __init__(self, img_lbl_tensor_dir,transform=None,preprocessing=None,device='cpu'):
        '''Initialization'''

        img_lbl_tensor_path = Path(img_lbl_tensor_dir)
        
        try:
            assert(img_lbl_tensor_path.is_dir())
        except:
            print(f"{img_lbl_tensor_path} is not a valid path!!\n")
        
        else:
            batches = list(img_lbl_tensor_path.glob('*.pt'))

            print(f"\nLoading Tensor Data Chunk {0}/{len(batches)}\n")

            img_data,lbl_data = load_img_lbl_tensors(batches[0],device=device)

            if transform:
                print("\nTransforming data.\n")
                img_data,lbl_data = transform((img_data,lbl_data))

            if preprocessing:
                print("\nPreprocessing data.\n")
                img_data,lbl_data = preprocessing((img_data,lbl_data))


            self.curr_data = (img_data,lbl_data)
            self.curr_batch = 0

            self.num_samples = len(img_data)*len(batches)
            self.batches = batches
            self.samp_per_batch = len(img_data)
            self.num_batches = len(batches)
            self.transform = transform
            self.preprocessing = preprocessing
            self.device = device
            #images,labels = load_img_lbl_tensors(img_lbl_tensor_path,device=device)
            #self.images = images
            #self.labels = labels
            #self.num_samples = len(images)
        
    def __len__(self):
        '''Returns total number of samples'''
        return self.num_samples

    def __getitem__(self, abs_idx):
        '''Generates a single sample of the data'''
        
        batch = math.floor(abs_idx/self.samp_per_batch)
        idx = abs_idx - (batch * self.samp_per_batch)

        #print(
        #    f"abs_idx: {abs_idx}\n"
        #    f"batch: {batch}\n"
        #    f"idx: {idx}\n"
        #)

        if self.curr_batch == batch:
            pass
        else:
            print(f"\nLoading Tensor Data Chunk {batch+1}/{self.num_batches}\n")
            #print(
            #    f"Current memory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            del self.curr_data
            gc.collect
            torch.cuda.empty_cache()
            self.curr_batch = batch
            self.curr_data = None

            img_lbl_tensor_f = self.batches[batch]
            img_lbl_tensor_path = Path(img_lbl_tensor_f)

            #print(
            #    f"Prior to loading next batch\nMemory allocated and cached: {torch.cuda.memory_allocated()}, {torch.cuda.memory_reserved()}\n"
            #)

            img_data,lbl_data = load_img_lbl_tensors(img_lbl_tensor_path,device=self.device)
            
            if self.transform:
                print("\nTransforming data.\n")
                img_data,lbl_data = self.transform((img_data,lbl_data))

            if self.preprocessing:
                print("\nPreprocessing data.\n")
                img_data,lbl_data = self.preprocessing((img_data,lbl_data))  

            self.curr_data = (img_data,lbl_data)

        image,label = self.curr_data[0][idx],self.curr_data[1][idx]
            
        return image,label