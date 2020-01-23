import SegAN_arch as seganorig
import SegAN_IO_arch as seganio
import DeepMRI as deepmri

dataset = {
           'training':'brats2015_training_crop_mri',
           'validation':'brats2015_validation_crop_mri',
           'testing':'brats2015_testing_crop_mri'
          }
modelname = 'SegAnIO_brats2015_ALL'
architecture=seganio
input_modalities = ["MR_T1", "MR_T1c", "MR_T2", "MR_Flair"]
output_labels = 4
model_input_size = 160
tracked_metric='dice_score_complete_tumor'
seed=1234567890


gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
gan.load_dataset(dataset, input_modalities)
gan.build_model(seed=seed, arch=architecture)
gan.train(tracked_metric=tracked_metric)