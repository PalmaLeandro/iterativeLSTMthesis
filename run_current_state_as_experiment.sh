# script made to use current repository state in an existing
# branch doing a commit with specific identifier in order to
# do easier the search of this state

# arg1: repository branch to checkout
# arg2: identifier of the state
# arg3: experiment to run
# arg4: model to choose

sudo git add -A

sudo git checkout $1

sudo git commit -m "\"$2\""

if [ "$#" -eq 4 ];then
	echo "Attempt to instantiate a new model"
	sudo python $3 --data_path=`pwd`/data/simple-examples/data/ --model $4 --logdir `pwd`/logs/$2 --erase --exportmodeldir `pwd`/model/$2 
fi
if [ "$#" -eq 5 ];then
	echo "Attempt to import existing model"
	sudo python $3 --data_path=`pwd`/data/simple-examples/data/ --model $4 --logdir `pwd`/logs/$2 --erase --exportmodeldir `pwd`/model/$2  --importmodeldir $5
fi