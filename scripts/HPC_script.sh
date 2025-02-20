
#!/bin/sh 
### General options  
### -- specify queue -- 
#BSUB -q gpua100
### -- set the job Name -- 
#BSUB -J First_submission_GNN
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s185395@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# Activate conda enviorment and load modules 
module load python3/3.9.19
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
source /zhome/be/1/138857/EHR_sensor_representations/venv/bin/activate

# Execute command
python cli.py --output_path="/zhome/be/1/138857/EHR_sensor_representations/output" --model_type=grud --epochs=100 --batch_size=32 --lr=0.0001 --recurrent_dropout=0.2 --recurrent_n_units=128
