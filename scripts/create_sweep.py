import wandb

sweep_config = {
    "program": "/zhome/be/1/138857/EHR_sensor_representations/scripts/cli.py",
    "method": "random",  # Search method: 'grid', 'random', or 'bayes'
    "metric": {"name": "final_auroc", "goal": "maximize"},  # Optimize for AUROC
    "parameters": {
        "model_type": {"values": ["grud"]},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [200]},
        "lr": {"values": [0.01, 0.001, 0.0001]},
        "recurrent_dropout": {"values": [0.2, 0.4, 0.6]},
        "recurrent_n_units": {"values": [64, 128, 256]},
        "expand_features": {"values": ["embeddings_pca_2"]},
        "patience": {"values": [10]},
        "wandb_sweep": {"values": [True]},
        "output_path": {"values": ["/work3/s185395/output/run_test_embeddings_pca_bert"]}
    },
}

# ✅ Create the sweep in W&B
sweep_id = wandb.sweep(sweep_config, project="test_sweep")
print(f"Sweep ID: {sweep_id}")

