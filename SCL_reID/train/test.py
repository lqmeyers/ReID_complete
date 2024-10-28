import torchvision
from transformers import ViTModel
from transformers import ViTFeatureExtractor, AutoImageProcessor
import sys


sys.path.insert(0,"../")

from utils.pytorch_data import *
from models.pytorch_models import *

# print('Getting ViT feature extractor...')
model_name = 'google/vit-base-patch16-224-in21k'
model_path = "/home/lmeyers/contrastive_learning_new_training/64_ids_batch1_sample_num_64/64_ids_batch1_sample_num_64.pth"

feature_extractor = AutoImageProcessor.from_pretrained(model_name)

model = torch.load(model_path)
# model = ViTModel.from_pretrained(model_name)
# model.to(device)



