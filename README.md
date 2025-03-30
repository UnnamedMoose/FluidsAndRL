# FluidsAndRL

## DONE
+ write a sbl eval script
+ add callback to sbl serial training
+ create a better file structure for training outputs in serial
+ add multithreading/GPU control to the env
+ fix process.wait() hang up
+ replace the kill function with what's in the env
+ link the new async data collection loop with the right sim
+ wrap the async data collection code into a self-contained function

## TODO
| change hyperparams to get a well-converged and functional set up for serial
- add a training loop from v0
- run full training with the asynchronous algo and same agent hyperparameters as for serial
- save figures and clean up the repo
- get the deterministic swimmer sim to work on a GPU without comms first
- make sure that can run comms with GPU
- rerun both trainings on GPU to be sure
- find a way to set the random seed in Julia for 1:1 evaluation checks
