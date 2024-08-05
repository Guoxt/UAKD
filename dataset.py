# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import PIL.Image
import torch
from torch.utils import data
from utils import Config


config = Config()


class MyData(data.Dataset):  # inherit
    """
    load data in a folder
    """

    def __init__(self, root, DF, transform=True):
        super(MyData, self).__init__()
        self.root = root
        self._transform = transform
        self.scale_size = config.SCALE_SIZE

        self.DF = pd.DataFrame(columns=['imgName', 'maskName', 'label', 'center', 'xmin', 'ymin',
                                        'xmax', 'ymax', 'width', 'height','discFlag','rater'])
        for spilt in DF:
            DF_all = pd.read_csv(root + '/' + 'Glaucoma_multirater_' + spilt + '.csv', encoding='gbk')

            DF_this = DF_all.loc[DF_all['rater'] == 0]      # Final Label
            DF_this = DF_this.reset_index(drop=True)
            DF_this = DF_this.drop('Unnamed: 0', 1)
            self.DF = pd.concat([self.DF, DF_this])

        self.DF.index = range(0, len(self.DF))


    def __len__(self):
        return len(self.DF)

    def __getitem__(self, index):
        img_Name = self.DF.loc[index, 'imgName']

        """ Get the images """
        fullPathName = os.path.join(self.root, img_Name)
        fullPathName = fullPathName.replace('\\', '/')  # image path

        img = PIL.Image.open(fullPathName).convert('RGB')  # read image
        img = img.resize((self.scale_size, self.scale_size))
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)    # add additional channel in dim 2 (channel)

        img_ori = img

        """ Get the six raters masks """
        mask_cup = []
        mask_disc = []
        #print(img_ori.shape)
        data_path = self.root
        for n in range(1,7):     # n:1-6
            # # load rater 1-6 label recurrently

            maskName = self.DF.loc[index, 'maskName'].replace('FinalLabel','Rater'+str(n))
            fullPathName = os.path.join(data_path, maskName)
            fullPathName = fullPathName.replace('\\', '/')

            Mask = PIL.Image.open(fullPathName).convert('L')
            Mask = Mask.resize((self.scale_size, self.scale_size))
            Mask = np.array(Mask)

            if Mask.max() > 1:
                Mask = Mask / 255.0
            
            disc = Mask.copy()
            disc[disc != 0] = 1
            cup = Mask.copy()
            cup[cup != 1] = 0
            #Mask = np.stack((disc,cup))
            
#             Mask = np.zeros((self.scale_size, self.scale_size))
#             Mask[disc==1] = 1
#             Mask[cup==1] = 2
            #np.save('userhome/GUOXUTAO/2022_31/67/RIGA_Baseline/00/mask.npy',Mask)
    
            mask_cup.append(cup)
            mask_disc.append(disc)

            # Mask = Mask.transpose((2, 0, 1))
        mask_cup = torch.from_numpy(np.array(mask_cup)).long()
        mask_disc = torch.from_numpy(np.array(mask_disc)).long()
        masks = []
        masks.append(mask_cup)
        masks.append(mask_disc)

        if self._transform:
            img_ori, img, masks = self.transform(img_ori, img, masks)
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}
        else:
            return {'image': img, 'image_ori': img_ori, 'mask': masks, 'name': img_Name.split('.')[0]}


    # Translating numpy_array into format that pytorch can use on Code.
    def transform(self, img_o, img, lbl):
        if img.max() > 1:
            img = img.astype(np.float32) / 255.0
        img -= config.MEAN_AND_STD['mean_rgb']
        img /= config.MEAN_AND_STD['std_rgb']
        img = img.transpose(2, 0, 1)  # to verify
        img = torch.from_numpy(img)

        if img.max() > 1:
            img_o = img_o.astype(np.float32) / 255.0
        img_o = img_o.transpose(2, 0, 1)  # to verify
        img_o = torch.from_numpy(img_o)

        
        return img_o, img, lbl
