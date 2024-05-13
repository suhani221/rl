
# Maternal health/
expname="maternal_A"
model="inputs/model_data/maternal_A.csv"
params="inputs/params/maternal_params.csv"
argsin="--expname ${expname} --model ${model} --params ${params}"
python run_experiments.py $argsin --n_chunks 8 > tmp/run.sh
echo wait >> tmp/run.sh
bash tmp/run.sh
python combine_results.py $argsin
python analysis.py $argsin --csv



