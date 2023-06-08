import torch

# Definir semente usada em operações como embaralhamento do dataset, pelo keras e TF
SEED = 42

DEVICE = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
