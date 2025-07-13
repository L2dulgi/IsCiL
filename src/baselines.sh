export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false


id=HelloIsCiL
env=kitchen # kitchen or mmworld(multi-stage metaworld)
seed=0 # any integer
type=complete # complete / semi / incomplete

python src/l2m_trainer.py --algo tail -id "$id"_"$type"_"$seed" -e $env -sp $type -epoch 5000 --seed $seed
python src/l2m_trainer.py --algo tailg -id "$id"_"$type"_"$seed" -e $env -sp $type -epoch 5000 --seed $seed
python src/l2m_trainer.py --algo l2mg -id "$id"_"$type"_"$seed" -e $env -sp $type -epoch 5000 --seed $seed
python src/l2m_trainer.py --algo l2m -id "$id"_"$type"_"$seed" -e $env -sp $type -epoch 5000 --seed $seed
