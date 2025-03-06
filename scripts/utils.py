import torch 
import numpy as np 

def get_one_hot_features(data, device):
    
    N = data.shape[0]
    F = data.shape[1]
    T = data.shape[2]

    # Identity matrix for one-hot encoding (F, F)
    one_hot = torch.eye(F, device=device).flatten().unsqueeze(0).unsqueeze(2)
    one_hot = torch.cat([one_hot] * N, dim=0)
    one_hot = torch.cat([one_hot] * T, dim=2)

    # Create a corresponding mask (1s for all one-hot encoded features)
    one_hot_mask = torch.ones_like(one_hot, dtype=torch.float32, device=device)

    # Create a corresponding delta (0s since one-hot encoding does not change over time)
    one_hot_delta = torch.zeros_like(one_hot, dtype=torch.float32, device=device)

    return one_hot, one_hot_mask, one_hot_delta

def get_embedding_features(data, device):
    
    N = data.shape[0]
    F = data.shape[1]
    T = data.shape[2]

    embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/small_embeddings.npy")

    # Change dimensions of embeddings 
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device).flatten().unsqueeze(0).unsqueeze(2)
    embeddings = torch.cat([embeddings] * N, dim=0)
    embeddings = torch.cat([embeddings] * T, dim=2)

    # Create a corresponding mask (1s for all one-hot encoded features)
    embeddings_mask = torch.ones_like(embeddings, dtype=torch.float32, device=device)

    # Create a corresponding delta (0s since one-hot encoding does not change over time)
    embeddings_delta = torch.zeros_like(embeddings, dtype=torch.float32, device=device)

    return embeddings, embeddings_mask, embeddings_delta