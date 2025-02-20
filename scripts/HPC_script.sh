
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
#BSUB -W 48:00 
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
source /zhome/be/1/138857/miniconda3/etc/profile.d/conda.sh
conda activate proteinworkshop
module load gcc/13.1.0-binutils-2.40

# Execute command
python3 main.py --mode train --run /dtu/blackhole/09/138857/special_project/results/run_378/
python3 main.py --mode test --run /dtu/blackhole/09/138857/special_project/results/run_378/
