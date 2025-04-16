from mortality_part_preprocessing import load_pad_separate
from mortality_classification import train_test
import os
import click
import torch
import random
import numpy as np
import json
import wandb


@click.command()
@click.option(
    "--output_path",
    default="./ehr_classification_results/",
    help="Path to output folder",
)
@click.option("--pooling", default="max", help="pooling function")
@click.option("--epochs", default=300, help="model dropout rate")
@click.option("--dropout", default=0.4, help="model dropout rate")
@click.option("--attn_dropout", default=0.4, help="model attention dropout rate")
@click.option(
    "--model_type", default="transformer", help="model_type"
)
@click.option("--heads", default=1, help="number of attention heads")
@click.option("--batch_size", default=64, help="batch size")
@click.option("--layers", default=1, help="number of attention layers")
@click.option("--dataset_id", default="physionet2012", help="filename id of dataset")
@click.option("--base_path", default="./P12data", help="Path to data folder")
@click.option("--lr", default=0.001, help="learning rate")
@click.option("--patience", default=10, help="patience for early stopping")
@click.option(
    "--use_mask",
    default=False,
    help="boolean, use mask for timepoints with no measurements",
)
@click.option(
    "--early_stop_criteria",
    default="auroc",
    help="what to early stop on. Options are: auroc, auprc, auprc+auroc, or loss",
)
@click.option("--seft_n_phi_layers", default=3)
@click.option("--seft_phi_width", default=32)
@click.option("--seft_phi_dropout", default=0.)
@click.option("--seft_n_psi_layers", default=2)
@click.option("--seft_psi_width", default=64)
@click.option("--seft_psi_latent_width", default=128)
@click.option("--seft_dot_prod_dim", default=128)
@click.option("--seft_latent_width", default=128)
@click.option("--seft_n_rho_layers", default=3)
@click.option("--seft_rho_width", default=32)
@click.option("--seft_rho_dropout", default=0.)
@click.option("--seft_max_timescales", default=100)
@click.option("--seft_n_positional_dims", default=4)
@click.option("--ipnets_imputation_stepsize", default=0.25)
@click.option("--ipnets_reconst_fraction", default=0.25)
@click.option("--recurrent_dropout", default=0.3)
@click.option("--recurrent_n_units", default=100)
@click.option("--expand_features", default=None)
@click.option("--wandb_sweep", default=False)

def core_function(
    output_path,
    base_path,
    model_type,
    epochs,
    dataset_id,
    batch_size,
    lr,
    patience,
    early_stop_criteria,
    expand_features, 
    wandb_sweep,
    **kwargs
):

    model_args = kwargs

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    accum_loss = []
    accum_accuracy = []
    accum_auprc = []
    accum_auroc = []
    
    if wandb_sweep:
        wandb.init(resume=True)  # Initialize W&B only for sweeps
        config = wandb.config  # Get sweep-configured hyperparameters
        epochs = config.epochs
        batch_size = config.batch_size
        lr = config.lr
        patience = config.patience
        kwargs.update(vars(config))

    split_range = range(1, 2) if wandb_sweep else range(1, 6) # only runs first split when sweeping

    for split_index in split_range:
        base_path_new = f"{base_path}/split_{split_index}"
        train_pair, val_data, test_data = load_pad_separate(
            dataset_id, base_path_new, split_index
        )

        # make necessary folders
        # if new model, make model folder
        if os.path.exists(output_path):
            pass
        else:
            try:
                os.mkdir(output_path)
            except OSError as err:
                print("OS error:", err)
        # make run folder
        base_run_path = os.path.join(output_path, f"split_{split_index}")
        run_path = base_run_path
        if os.path.exists(run_path):
        #    raise ValueError(f"Path {run_path} already exists.")
             print(f"{run_path} already exists, overwriting for now") #change to log in wandb
        #os.mkdir(run_path)
        os.makedirs(run_path, exist_ok = True) #REMOVE THIS EXIST_OK, IS A SOLUTION FOR NOW
        # save model settings
        model_settings = {
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "dataset": dataset_id,
            "learning_rate": lr,
            "patience": patience,
            "early_stop_criteria": early_stop_criteria,
            "base_path": base_path,
            "expand_features": expand_features,
            "pooling_fxn": model_args["pooling"],
        }
        if model_type == "transformer":
            model_settings["layers"] = model_args["layers"]
        if model_type in ("seft", "transformer"):
            model_settings["dropout"] = model_args["dropout"]
            model_settings["attn_dropout"] = model_args["attn_dropout"]
            model_settings["use_timepoint_mask"] = model_args["use_mask"]
            model_settings["heads"] = model_args["heads"]
        if model_type == "seft":
            model_settings["seft_n_phi_layers"] = model_args["seft_n_phi_layers"]
            model_settings["seft_phi_width"] = model_args["seft_phi_width"]
            model_settings["seft_phi_dropout"] = model_args["seft_phi_dropout"]
            model_settings["seft_n_psi_layers"] = model_args["seft_n_psi_layers"]
            model_settings["seft_psi_width"] = model_args["seft_psi_width"]
            model_settings["seft_psi_latent_width"] = model_args["seft_psi_latent_width"]
            model_settings["seft_dot_prod_dim"] = model_args["seft_dot_prod_dim"]
            model_settings["seft_latent_width"] = model_args["seft_latent_width"]
            model_settings["seft_n_rho_layers"] = model_args["seft_n_rho_layers"]
            model_settings["seft_rho_width"] = model_args["seft_rho_width"]
            model_settings["seft_rho_dropout"] = model_args["seft_rho_dropout"]
        if model_type in ("grud", "ipnets"):
            model_settings["recurrent_dropout"] = model_args["recurrent_dropout"]
            model_settings["recurrent_n_units"] = model_args["recurrent_n_units"]
            model_settings["expand_features"] = expand_features
        if model_type == "ipnets":
            model_settings["ipnets_imputation_stepsize"] = model_args["ipnets_imputation_stepsize"]
            model_settings["ipnets_reconst_fraction"] = model_args["ipnets_reconst_fraction"]

        with open(f"{run_path}/model_settings.json", "w") as fp:
            json.dump(model_settings, fp)

        # run training
        loss, accuracy_score, auprc_score, auc_score = train_test(
            train_pair,
            val_data,
            test_data,
            output_path=run_path,
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            early_stop_criteria=early_stop_criteria,
            expand_features=expand_features,
            #wandb_sweep=wandb_sweep,
            model_args=model_args,
        )

        accum_loss.append(loss)
        accum_accuracy.append(accuracy_score)
        accum_auprc.append(auprc_score)
        accum_auroc.append(auc_score)

        if wandb_sweep:
            # Log final metrics at the end of each run
            wandb.log({
                "final_loss": loss,
                "final_accuracy": accuracy_score,
                "final_auprc": auprc_score,
                "final_auroc": auc_score
            })

            # Ensure W&B run is properly closed
            wandb.finish()
    with open(f"{output_path}/summary.json", "w") as f:
        json.dump(
            {
                "mean_loss": float(np.mean(accum_loss)),
                "mean_accuracy": float(np.mean(accum_accuracy)),
                "mean_auprc": float(np.mean(accum_auprc)),
                "mean_auroc": float(np.mean(accum_auroc)),
                "std_loss": float(np.std(accum_loss)),
                "std_accuracy": float(np.std(accum_accuracy)),
                "std_auprc": float(np.std(accum_auprc)),
                "std_auroc": float(np.std(accum_auroc)),
            }, f, indent=4,
        )

def sweep_train():
    core_function(
        output_path="./ehr_classification_results/",
        base_path="./P12data",
        model_type="grud",  # Default, but W&B will override it
        epochs=5,  # Default, but W&B will override it
        dataset_id="physionet2012",
        batch_size=32,  # Default, but W&B will override it
        lr=0.0001,  # Default, but W&B will override it
        patience=10,
        early_stop_criteria="auroc",
        expand_features=None,
        wandb_sweep=True  # Important to enable W&B sweep mode
    )
if __name__ == "__main__":
    if wandb.run:  # ✅ Detects if W&B is controlling execution (for sweeps)
        sweep_train()  # Runs training with hyperparameters from W&B
    else:
        core_function()  # Runs normal training when executed manually

