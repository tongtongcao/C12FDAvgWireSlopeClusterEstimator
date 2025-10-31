The package is to train avergage-wire and slope estimator for missing cluster on a track by Transformer Masked Auto-Encoder (TMAE).\
The AI model is designed by pytorch and pytorch-lightning.

It is recommended to run the package in python virtual enviroment.
  - To install python virtual enviromen for cup-only, source setup_pytorch_lightning_venv.csh
  - To install python virtual enviromen for GPU, cuda needs to be installed. In ifarm, cuda 12.4 has been installed. Change "set use_cuda = 0" into "set use_cuda = 1", then source setup_pytorch_lightning_venv.csh. Python virtual enviromen for GPU will be installed.

Then, enter python virtual enviroment,
  - source venvs/torch-cpu/bin/activate.csh for cpu-only
  - source venvs/torch-cu124/bin/activate.csh for GPU at ifarm
  
Finally, run the package,
- python3 train.py ...

Arguments:
  - positional arguments:
    - inputs      &nbsp;&nbsp;&nbsp;          One or more CSV files (default: avgWires.csv)

  - options:
    - -h, --help     &nbsp;&nbsp;&nbsp;       show this help message and exit
    - --device {cpu,gpu,auto} &nbsp;&nbsp;&nbsp; Choose device: cpu, gpu, or auto (default: auto)
    - --max_epochs MAX_EPOCHS &nbsp;&nbsp;&nbsp; Number of training epochs
    - --batch_size BATCH_SIZE &nbsp;&nbsp;&nbsp; Batch size for DataLoader
    - --outdir OUTDIR   &nbsp;&nbsp;&nbsp;    Directory to save models and plots
    - --end_name END_NAME &nbsp;&nbsp;&nbsp;  Optional suffix to append to output files (default: none)
    - --d_model D_MODEL  &nbsp;&nbsp;&nbsp;   Transformer embedding dimension (must be divisible by nhead)
    - --nhead NHEAD   &nbsp;&nbsp;&nbsp;      Number of attention heads in the transformer (default: 4)
    - --num_layers NUM_LAYERS &nbsp;&nbsp;&nbsp; Number of transformer encoder layers (default: 2)
    - --lr LR        &nbsp;&nbsp;&nbsp;       Learning rate for optimizer (default: 1e-3)
    - --no_train      &nbsp;&nbsp;&nbsp;      Skip training and only run inference using a saved model
    - --enable_progress_bar &nbsp;&nbsp;&nbsp; Enable progress bar during training (default: disabled)

For the estimator, inputs are two paramters:
  1. Sequence of average wires for other 5 clusters
  2. Index of missing cluster ( = superlayer - 1)
     
and outputs are average wire and slope of missing cluster.

To apply the estimator in coatjava, an example for application of the estimator by ai.djl is developed.
- Install: mvn clean install
- Run: java -cp "target/TestDJL-1.0-SNAPSHOT.jar:target/lib/*" org.example.Main
