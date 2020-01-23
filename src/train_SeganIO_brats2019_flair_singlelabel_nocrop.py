import SegAN_arch as seganorig
import SegAN_IO_arch as seganio
import DeepMRI as deepmri

dataset = {
           'training':'brats2019_training_mri',
           'validation':'brats2019_validation_mri',
           'testing':'brats2019_testing_mri'
          }

modelname = 'SegAnIO_brats2019_flair_singlelab_nocrop'
architecture=seganio
input_modalities = ["flair"]
output_labels = 1
model_input_size = 192
tracked_metric='dice_score_0'
seed=1234567890


gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
gan.load_dataset(dataset, input_modalities, training_random_crop=False, training_center_crop=True)
gan.build_model(seed=seed, arch=architecture)
gan.train(tracked_metric=tracked_metric)