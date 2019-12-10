import SegAN_arch as seganorig
import SegAN_IO_arch as seganio
import DeepMRI as deepmri

dataset = {
           'training':'brats2019_training_crop_mri',
           'validation':'brats2019_validation_crop_mri',
           'testing':'brats2019_testing_crop_mri'
          }

modelname = 'SegAnIO_brats2019_ALL_multilab'
architecture=seganio
input_modalities = ["t1", "t1ce", "t2", "flair"]
output_labels = 4
model_input_size = 160
tracked_metric='dice_score_complete_tumor_2019'
seed=1234567890


gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
gan.load_dataset(dataset, input_modalities)
gan.build_model(seed=seed, arch=architecture)
gan.train(tracked_metric=tracked_metric)