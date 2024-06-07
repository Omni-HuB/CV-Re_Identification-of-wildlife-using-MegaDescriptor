# %%


# %%
!pip install wildlife-datasets





# %%


# %%
pip install wildlife-tools

# %%
pip install wildlife-datasets


# %%
pip install timm

# %% [markdown]



# %% [markdown]
# 

# %%
from itertools import chain
import timm
import pandas as pd
import torchvision.transforms as T
from torch.optim import SGD

from wildlife_tools.data import WildlifeDataset, SplitMetadata
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

import timm
import numpy as np
from wildlife_datasets.datasets import MacaqueFaces
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
from wildlife_datasets import datasets, splits
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# %%
import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

# Download dataset (if not already downloaded)
datasets.IPanda50.get_data('../data/IPanda50')
# Load dataset metadata
metadata = datasets.IPanda50('../data/IPanda50')


transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset = WildlifeDataset(metadata.df, metadata.root, transform=transform)


# Download MegaDescriptor-T backbone from HuggingFace Hub
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

# Arcface loss - needs backbone output size and number of classes.
objective = ArcFaceLoss(
    num_classes=dataset.num_classes,
    embedding_size=768,
    margin=0.5,
    scale=64
    )

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

def print_epoch_loss(trainer, epoch_data):
    # This function will print the average loss at the end of each epoch
    print(f"Epoch {trainer.epoch}: Average Loss = {epoch_data['train_loss_epoch_avg']}")


trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=2,
    device='cuda',
    epoch_callback=print_epoch_loss
)

trainer.train()


# # %%

# trainer.save("retrained_chks", file_name="arcfaceloass_retrained_checkpoint.pth")



# %%
dataset_database_P = WildlifeDataset(metadata.df.iloc[100:,:], metadata.root, transform=transform)
dataset_query_P = WildlifeDataset(metadata.df.iloc[:100,:], metadata.root, transform=transform)

# %%
# name = 'hf-hub:BVRA/MegaDescriptor-T-224'
extractor_P = DeepFeatures(trainer.model)
query_P, database_P = extractor_P(dataset_query_P), extractor_P(dataset_database_P)

# %%


# %%
similarity_function = CosineSimilarity()
similarity_P = similarity_function(query_P, database_P)
print(similarity_P)

# %%
classifier_P = KnnClassifier(k=1, database_labels=dataset_database_P.labels_string)
predictions_P = classifier_P(similarity_P['cosine'])
print("Predictions for 100 test Images:-\n",predictions_P)
accuracy_P = np.mean(dataset_query_P.labels_string == predictions_P)
print("Accuracy on IPanda50 data: {:.2f}%".format(accuracy_P * 100))
precision_P = precision_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
recall_P = recall_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
f1_P = f1_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
print("Precision:", precision_P)
print("Recall:", recall_P)
print("F1 Score:", f1_P)











##################################################################################

# %% [markdown]
# TRIPLET_LOSS

# %%
from itertools import chain
import timm
import pandas as pd
import torchvision.transforms as T
from torch.optim import SGD

from wildlife_tools.data import WildlifeDataset, SplitMetadata
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

import timm
import numpy as np
from wildlife_datasets.datasets import MacaqueFaces
from wildlife_tools.data import WildlifeDataset
import torchvision.transforms as T
from wildlife_datasets import datasets, splits
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

import timm
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer , TripletLoss

# Download dataset (if not already downloaded)
datasets.IPanda50.get_data('../data/IPanda50')
# Load dataset metadata
metadata = datasets.IPanda50('../data/IPanda50')


transform = T.Compose([T.Resize([224, 224]), T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
dataset = WildlifeDataset(metadata.df, metadata.root, transform=transform)


# Download MegaDescriptor-T backbone from HuggingFace Hub
backbone = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True)

# Arcface loss - needs backbone output size and number of classes.
objective = TripletLoss()

# Optimize parameters in backbone and in objective using single optimizer.
params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

def print_epoch_loss(trainer, epoch_data):
    # This function will print the average loss at the end of each epoch
    print(f"Epoch {trainer.epoch}: Average Loss = {epoch_data['train_loss_epoch_avg']}")


trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=2,
    device='cuda',
    epoch_callback=print_epoch_loss
)

trainer.train()



# # %%
# import torch

# trainer.save("retrained_chks", file_name="tripletloss_IPanda50_retrained_checkpoint.pth")




# %%
dataset_database_P = WildlifeDataset(metadata.df.iloc[100:,:], metadata.root, transform=transform)
dataset_query_P = WildlifeDataset(metadata.df.iloc[:100,:], metadata.root, transform=transform)

# name = 'hf-hub:BVRA/MegaDescriptor-T-224'
extractor_P = DeepFeatures(trainer.model , device = 'cuda')

query_P, database_P = extractor_P(dataset_query_P), extractor_P(dataset_database_P)





# %%
similarity_function = CosineSimilarity()
similarity_P = similarity_function(query_P, database_P)
print(similarity_P)


# %%
classifier_P = KnnClassifier(k=1, database_labels=dataset_database_P.labels_string)
predictions_P = classifier_P(similarity_P['cosine'])
print("Predictions for 100 test Images:-\n",predictions_P)
accuracy_P = np.mean(dataset_query_P.labels_string == predictions_P)
print("Accuracy on IPanda50 data: {:.2f}%".format(accuracy_P * 100))
precision_P = precision_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
recall_P = recall_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
f1_P = f1_score(dataset_query_P.labels_string, predictions_P, average='weighted',zero_division=1)
print("Precision:", precision_P)
print("Recall:", recall_P)
print("F1 Score:", f1_P)


