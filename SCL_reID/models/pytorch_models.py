import numpy as np
import torch
from torch import nn
import torchvision
from transformers import ViTModel
import torch.nn.functional as F

import sys
sys.path.insert(0, '../')
from models.pytorch_resnet50_conv3 import resnet50_convstage3
from models.pytorch_resnet50 import resnet50
import open_clip

###################################################################################################
#
# CODE TO BUILD PYTORCH MODELS
#
###################################################################################################


###################################################################################################
# MODEL WITH ViT BACKBONE FOR CLASSIFICATION
#
class ViTForClassification(nn.Module):

    def __init__(self, num_labels=10):
        super(ViTForClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.linear = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.linear(outputs['pooler_output'])
        return logits
###################################################################################################

###################################################################################################
# MODEL WITH ViT BACKBONE FOR REID
#
class ViTForReID(nn.Module):

    def __init__(self, latent_dim=128,load_saved=False,model_path=None):
        super(ViTForReID, self).__init__()
        if load_saved == False:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            self.vit = ViTModel.from_pretrained(model_path)
        #"""
        self.reid = nn.Sequential(#nn.Dropout(0.2),
                                  nn.Linear(self.vit.config.hidden_size, latent_dim),         
                                  #nn.Dropout(0.2))
        )
        #"""
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.vit.config.hidden_size, self.vit.config.hidden_size)
        self.fc2 = nn.Linear(self.vit.config.hidden_size, latent_dim)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # hidden = F.relu(self.fc1(outputs['pooler_output']))
        # embedding = self.fc2(hidden)
        embedding = self.reid(outputs['pooler_output'])
        embedding = embedding.div(torch.linalg.vector_norm(embedding))
        return embedding
###################################################################################################

###################################################################################################
# MODEL WITH ViT BACKBONE FOR REID
#
class CLIPForReID(nn.Module):

    def __init__(self, latent_dim=128,load_saved=False,model_path=None):
        super(CLIPForReID, self).__init__()
        self.bio_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.vit = self.bio_model.visual
        #clip_flattener = CLIPFeatureExtractor()
    
        #"""
        self.reid = nn.Sequential(#nn.Dropout(0.2),
                                  nn.Linear(self.vit.output_dim, latent_dim),      #supposedly this already has pooling in it   
                                  #nn.Dropout(0.2))
        )
        #"""
        self.latent_dim = latent_dim
        # self.fc1 = nn.Linear(self.vit.config.hidden_size, self.vit.config.hidden_size)
        # self.fc2 = nn.Linear(self.vit.config.hidden_size, latent_dim)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        logits = outputs[0]
        # hidden = F.relu(self.fc1(outputs['pooler_output']))
        # embedding = self.fc2(hidden)
        embedding = self.reid(logits)
        embedding = embedding.div(torch.linalg.vector_norm(embedding))
        return embedding
###################################################################################################





###################################################################################################
# MODEL WITH SWIN3D BACKBONE FOR TRACK CLASSIFICATION 
#
class SwinForTrackClassification(nn.Module):

    def __init__(self, num_labels=10):
        super(SwinForTrackClassification, self).__init__()
        #self.swin = torchvision.models.video.swin3d_s(weights=torchvision.models.video.Swin3D_S_Weights.KINETICS400_V1)
        #self.swin = torchvision.models.video.swin3d_b(weights=torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        self.swin = torchvision.models.video.swin3d_s()
        self.linear = nn.Linear(400, num_labels)
        self.num_labels = num_labels

    def forward(self, inputs):
        outputs = self.swin(inputs)
        logits = self.linear(outputs)
        return logits
###################################################################################################

###################################################################################################
# MODEL WITH SWIN3D BACKBONE FOR TRACK REID
#   
class SwinForTrackReID(nn.Module):

    def __init__(self, latent_dim=128):
        super(SwinForTrackReID, self).__init__()
        #self.swin = torchvision.models.video.swin3d_s(weights=torchvision.models.video.Swin3D_S_Weights.KINETICS400_V1)
        #self.swin = torchvision.models.video.swin3d_b(weights=torchvision.models.video.Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1)
        self.swin = torchvision.models.video.swin3d_s()
        self.reid = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(400, latent_dim),
                                  nn.Dropout(0.2))
        self.latent_dim = latent_dim

    def forward(self, inputs):
        outputs = self.swin(inputs)
        embedding = self.reid(outputs)
        embedding = embedding.div(torch.linalg.vector_norm(embedding))
        return embedding
###################################################################################################



###################################################################################################
# FUNCTION TO BUILD MODEL
#
def build_model(model_config):

    model_class = model_config['model_class']
    if model_class == 'resnet50_full':
        return resnet50()
    elif model_class == 'resnet50_conv3':
        return resnet50_convstage3()
    elif model_class == 'vit_classifier':
        return ViTForClassification(model_config['num_labels'])
    elif model_class == 'vit_reid':
        return ViTForReID(model_config['latent_dim'])
    elif model_class == 'clip_reid':
        return CLIPForReID(model_config['latent_dim'])
    elif model_class == 'swin3d_classifier':
        return SwinForTrackClassification(model_config['num_labels'])
    elif model_class == 'swin3d_reid':
        return SwinForTrackReID(model_config['latent_dim'])
        
    print(f'ERROR - requested model not available {model_class}')
    return -1
