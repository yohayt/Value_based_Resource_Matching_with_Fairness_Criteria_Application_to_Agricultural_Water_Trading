#!/bin/bash
function welfare(){    # requires dataset as an argument yakima/touchet 
data=$1
echo $data

for acre in `echo "5 10 20"`
do
    for delta in `seq 10 10 90`
    do
        log=out.${data}_${acre}_${delta}
        echo $log
sbatch -o $log \
    --mem=20G \
    --export=command="python ../scripts/experiment_welfare_real.py \
    -i $data \
    -d $delta \
    -a $acre" \
    ../scripts/run_proc.sbatch
    done
done
}

function summary(){
echo "network,data,delta,acre,num_edges,num_nodes,num_buyers,num_sellers,welfare,sellers_value_by_tot_value,sigma_T_by_sigma_0,sigma_0" > results_realdata.csv
tail -n1 out.* -q >> results_realdata.csv
}

if [[ $# == 0 ]]; then
   echo "Here are the options:"
   grep "^function" $BASH_SOURCE | sed -e 's/function/  /' -e 's/[(){]//g' -e '/IGNORE/d'
else
   eval $1 $2 $3
fi
