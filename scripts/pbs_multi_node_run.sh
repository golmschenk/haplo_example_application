#PBS -l select=2:model=mil_a100:ncpus=40:ngpus=4:mem=500GB
#PBS -l place=scatter:excl
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -W group_list=s2853
#PBS -q gpu_debug@pbspl4

# Logging setup.
session_name="session_name"
current_time=$(date "+%Y_%m_%d_%H_%M_%S")
export HAPLO_SESSION_DIRECTORY="sessions/${current_time}_${session_name}"
mkdir -p "${HAPLO_SESSION_DIRECTORY}"
qalter -o "${HAPLO_SESSION_DIRECTORY}/${PBS_JOBID}.log" $PBS_JOBID

# Handle special shells (not needed for many users).
source /usr/local/lib/init/global.profile

# Activate Conda environment.
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4
source activate haplo_env

# Setup MPI settings.
module load mpi-hpe/mpt
export MPI_SHEPHERD=true  # MPI is only being used to start the processes, not manage them.
export MPI_DSM_DISTRIBUTE=0  # Make sure MPI does not restrict CPU usage of unmanaged processes.
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Make MPI pass positional identifiers for GPUs, instead of UIDs.

head_node_hostname=`/bin/hostname -s`

mpiexec -perhost 1 python -m torch.distributed.run \
--nnodes 2 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_hostname \
scripts/example_train_session.py
