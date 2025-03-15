import wandb

sweep_config = {
    "program": "/zhome/be/1/138857/EHR_sensor_representations/scripts/cli.py",
    "method": "random",  # Search method: 'grid', 'random', or 'bayes'
    "metric": {"name": "final_auroc", "goal": "maximize"},  # Optimize for AUROC
    "parameters": {
        "model_type": {"values": ["grud"]},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [20,50, 100]},
        "lr": {"values": [0.0001, 0.001, 0.01]},
        "recurrent_dropout": {"values": [0.2, 0.4, 0.6]},
        "recurrent_n_units": {"values": [64, 128, 256]},
        "expand_features": {"values": [False]},
        "patience": {"values": [30]},
        "wandb_sweep": {"values": [True]},
        "output_path": {"values": ["/zhome/be/1/138857/EHR_sensor_representations/output/run_test"]}
    },
}

# ✅ Create the sweep in W&B
sweep_id = wandb.sweep(sweep_config, project="test_sweep")
print(f"Sweep ID: {sweep_id}")

