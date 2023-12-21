All code used for Federated Learning Model training was provided by Bo Li and adapted by us.

Requirements:
- torch
- torchvision

Furthermore, because Federated Learning is very computationally expensive, we ran CIFAR-10 experiments exclusively on DTU's 
HPC.

To run it on HPC, first clone the repository:

```
git clone https://github.com/aerte/DFL.git
```

Then create a virtual environment called `torch_dl` or change the `source` in the job script `submit_job` to your desired
environment.

Then just submit the job via:

```
bsub < submit_job.sh
```

All the relevant settings like the number of local epochs, alpha or what type of model to use (MLP,CNN or VGG11) can be 
changed in `run_cifar.sh`.

For post-processing of the results we used the three notebooks `DataBinning`, `DataPlots` and `ExamplePlots`. Please
refer to the notebooks and our report for insight into how the uncertainty estimation was conducted.
