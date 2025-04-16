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

# Trying a new function that for each measurment the sensor identity is concatenated after.
# Instead of after all measurments 
def get_one_hot_features_2(data, mask, delta):
    """
    Augments data, mask, and delta by adding per-feature one-hot identities.
    
    Inputs:
        data, mask, delta: tensors of shape (N, F, T)
        
    Returns:
        augmented_data, augmented_mask, augmented_delta: shape (N, F*(1+F), T)
    """
    N, F, T = data.shape
    device = data.device

    # Prepare output tensors
    out_dim = F * (1 + F)
    augmented_data = torch.zeros((N, out_dim, T), device=device)
    augmented_mask = torch.zeros_like(augmented_data)
    augmented_delta = torch.zeros_like(augmented_data)

    identity = torch.eye(F, device=device)  # (F, F)

    for f in range(F):
        base = f * (1 + F)

        # Insert measurement
        augmented_data[:, base, :] = data[:, f, :]
        augmented_mask[:, base, :] = mask[:, f, :]
        augmented_delta[:, base, :] = delta[:, f, :]

        # Insert one-hot vector for that feature
        one_hot_vec = identity[f]  # shape (F,)
        one_hot_expanded = one_hot_vec.view(1, F, 1).expand(N, F, T)  # (N, F, T)

        augmented_data[:, base + 1 : base + 1 + F, :] = one_hot_expanded
        augmented_mask[:, base + 1 : base + 1 + F, :] = 1.0
        augmented_delta[:, base + 1 : base + 1 + F, :] = 0.0

    return augmented_data, augmented_mask, augmented_delta


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

# Trying a new function that for each measurment the sensor identity is concatenated after.
# Instead of after all measurments 
def get_embedding_features_2(data, mask, delta):
    """
    Augment data with feature identity embeddings instead of one-hot.
    
    Args:
        data:      Tensor (N, F, T)
        mask:      Tensor (N, F, T)
        delta:     Tensor (N, F, T)
        embeddings: numpy array or tensor (F, D) — where D is embedding dim
        
    Returns:
        augmented_data, augmented_mask, augmented_delta: shape (N, F*(1+D), T)
    """
    embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/small_embeddings.npy")
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=data.device)

    N, F, T = data.shape
    D = embeddings.shape[1]  # embedding dim
    device = data.device

    block_size = 1 + D  # measurement + embedding vector
    out_dim = F * block_size

    augmented_data = torch.zeros((N, out_dim, T), device=device)
    augmented_mask = torch.zeros_like(augmented_data)
    augmented_delta = torch.zeros_like(augmented_data)

    for f in range(F):
        base = f * block_size

        # Insert measurement
        augmented_data[:, base, :] = data[:, f, :]
        augmented_mask[:, base, :] = mask[:, f, :]
        augmented_delta[:, base, :] = delta[:, f, :]

        # Insert embedding
        emb_vec = embeddings[f]  # (D,)
        emb_expanded = emb_vec.view(1, D, 1).expand(N, D, T)  # (N, D, T)

        augmented_data[:, base + 1 : base + 1 + D, :] = emb_expanded
        augmented_mask[:, base + 1 : base + 1 + D, :] = 1.0
        augmented_delta[:, base + 1 : base + 1 + D, :] = 0.0

    return augmented_data, augmented_mask, augmented_delta

def get_embedding_features_pca(data, mask, delta):
    """
    Augment data with feature identity embeddings instead of one-hot.
    
    Args:
        data:      Tensor (N, F, T)
        mask:      Tensor (N, F, T)
        delta:     Tensor (N, F, T)
        embeddings: numpy array or tensor (F, D) — where D is embedding dim
        
    Returns:
        augmented_data, augmented_mask, augmented_delta: shape (N, F*(1+D), T)
    """
    embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/pca_small_embeddings.npy")
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=data.device)

    N, F, T = data.shape
    D = embeddings.shape[1]  # embedding dim
    device = data.device

    block_size = 1 + D  # measurement + embedding vector
    out_dim = F * block_size

    augmented_data = torch.zeros((N, out_dim, T), device=device)
    augmented_mask = torch.zeros_like(augmented_data)
    augmented_delta = torch.zeros_like(augmented_data)

    for f in range(F):
        base = f * block_size

        # Insert measurement
        augmented_data[:, base, :] = data[:, f, :]
        augmented_mask[:, base, :] = mask[:, f, :]
        augmented_delta[:, base, :] = delta[:, f, :]

        # Insert embedding
        emb_vec = embeddings[f]  # (D,)
        emb_expanded = emb_vec.view(1, D, 1).expand(N, D, T)  # (N, D, T)

        augmented_data[:, base + 1 : base + 1 + D, :] = emb_expanded
        augmented_mask[:, base + 1 : base + 1 + D, :] = 1.0
        augmented_delta[:, base + 1 : base + 1 + D, :] = 0.0

    return augmented_data, augmented_mask, augmented_delta

def get_embedding_features_pca_2(data, mask, delta):
    """
    Augment data with feature identity embeddings instead of one-hot.
    
    Args:
        data:      Tensor (N, F, T)
        mask:      Tensor (N, F, T)
        delta:     Tensor (N, F, T)
        embeddings: numpy array or tensor (F, D) — where D is embedding dim
        
    Returns:
        augmented_data, augmented_mask, augmented_delta: shape (N, F*(1+D), T)
    """
    #embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/pca_2_small_embeddings.npy")
    #embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/pca_2_large_embeddings.npy")
    embeddings = np.load("/zhome/be/1/138857/EHR_sensor_representations/scripts/embeddings/pca_pubmedbert_embeddings.npy")
    embeddings = torch.tensor(embeddings, dtype=torch.float32, device=data.device)

    N, F, T = data.shape
    D = embeddings.shape[1]  # embedding dim
    device = data.device

    block_size = 1 + D  # measurement + embedding vector
    out_dim = F * block_size

    augmented_data = torch.zeros((N, out_dim, T), device=device)
    augmented_mask = torch.zeros_like(augmented_data)
    augmented_delta = torch.zeros_like(augmented_data)

    for f in range(F):
        base = f * block_size

        # Insert measurement
        augmented_data[:, base, :] = data[:, f, :]
        augmented_mask[:, base, :] = mask[:, f, :]
        augmented_delta[:, base, :] = delta[:, f, :]

        # Insert embedding
        emb_vec = embeddings[f]  # (D,)
        emb_expanded = emb_vec.view(1, D, 1).expand(N, D, T)  # (N, D, T)

        augmented_data[:, base + 1 : base + 1 + D, :] = emb_expanded
        augmented_mask[:, base + 1 : base + 1 + D, :] = 1.0
        augmented_delta[:, base + 1 : base + 1 + D, :] = 0.0

    return augmented_data, augmented_mask, augmented_delta
