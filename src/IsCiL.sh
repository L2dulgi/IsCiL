export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false


id=HelloIsCiL
env=kitchen # kitchen or mmworld(multi-stage metaworld)
seed=0 # any integer
type=complete # complete / semi / incomplete

python src/l2m_trainer.py --algo iscil -k 20 -id "$id"_"$type"_"$seed" -r 4 -lr 5e-4 -e $env -sp $type -epoch 5000 --seed $seed
