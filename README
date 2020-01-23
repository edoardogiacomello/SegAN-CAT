### SegAN-CAT

### Environment setup instructions
The project is provided as a repository of python scripts and a pre-configured docker environment running Jupyter Lab.

For running the environment with GPU support you will need nvidia-docker, cuda=10.0

1 - Clone the repo
2 - Within the repo root folder launch the script  ``` build_docker_[cpu|gpu].sh ```  according to your gpu availability. This will build a docker image tagged  ``` edoardogiacomello/deepmri2:latest-[gpu|cpu] ``` 
3 - For running the environment for the first time, launch the  ``` script first_run_[cpu|gpu].sh ```  according to your gpu availability. This will run a container named deepmri2.
4 - run  ``` tmux ```  for managing multiple bash sessions.
5 - Run Jupyter lab with ``` jupyter lab --ip=0.0.0.0 --allow-root ``` 

Jupyter will be available at <your-ip>:8887. You can optionally run tensorboard on port 6006.

Use  ``` ctrl+p+q ```  to detach from the docker environment. Use ``` docker attach deepmri2 ``` to re-attach to the environment.

### Paths description
- `train_<arch>_<dataset>_options.py`: Scripts for training the base models
- `transfer_<arch>_<dataset>_options.py`: Scripts for performing transfer learning. Edit your script to your needs
- `dataset_helpers.py`: Helper functions for managing the dataset (both conversion/pre-processing from raw to .tfrecords and to load the dataset during training/evaluation)
- `DeepMRI.py` main class that implements the training/validation/testing process.
- `SegAN_IO_arch.py`definition of our proposed architecture to be used in `DeepMRI.py`
- `SegAN_no_dice_arch.py`definition of the baseline architecture to be used in `DeepMRI.py`
- `SegAN_arch.py`definition of the SegAN \w Dice Loss architecture to be used in `DeepMRI.py`
- `TestScores.ipynb` Notebook for evaluating the networks.
- `DatasetPreprocessing.ipynb`: Example of how to run the pre-processing to convert from .mha to .tfrecords
- `MajorityVoting*.ipynb` Notebook for performing majority voting between multiple models
- `NegotiationTools.py` class needed for Majority voting experiments 
- `PredictOnChallenge.ipynb` notebook for end-to-end prediction from .mha files. Saves the results in the brats2015 name format
- `/models/<run_name>/`: This folders will be created upon the first training. It contains the checkpoints, a csv for training and validation steps and the tensorboard data for each run.
