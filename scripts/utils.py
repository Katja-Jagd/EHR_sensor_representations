import torch 

def get_one_hot_features(num_features, batch_size, time_steps, device):
    """
    Creates a one-hot encoding matrix for all features and generates corresponding mask and delta.

    Args:
    - num_features (int): Number of original features in X.
    - batch_size (int): Number of samples in the batch.
    - time_steps (int): Number of time steps (sequence length).
    - device (str): CUDA or CPU device.

    Returns:
    - feature_one_hot (torch.Tensor): One-hot encoded feature matrix.
    - one_hot_mask (torch.Tensor): Corresponding mask for one-hot features.
    - one_hot_delta (torch.Tensor): Corresponding delta values (zero since one-hot is static).
    """

    # Identity matrix for one-hot encoding (F, F)
    feature_one_hot = torch.eye(num_features, device=device)  # Shape: (F, F)

    # Expand to match batch and time dimensions
    feature_one_hot = feature_one_hot.unsqueeze(0).unsqueeze(2)  # Shape: (1, F, 1, F)
    feature_one_hot = feature_one_hot.expand(batch_size, -1, time_steps, -1)  # Shape: (B, F, T, F)

    # Reshape for concatenation with data (flatten the one-hot encoding)
    feature_one_hot = feature_one_hot.reshape(batch_size, num_features * num_features, time_steps)  # (B, F*F, T)

    # Create a corresponding mask (1s for all one-hot encoded features)
    one_hot_mask = torch.ones_like(feature_one_hot, dtype=torch.float32, device=device)  # (B, F*F, T)

    # Create a corresponding delta (0s since one-hot encoding does not change over time)
    one_hot_delta = torch.zeros_like(feature_one_hot, dtype=torch.float32, device=device)  # (B, F*F, T)

    return feature_one_hot, one_hot_mask, one_hot_delta