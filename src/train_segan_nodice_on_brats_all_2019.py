#import SegAN_arch as seganorig
#import SegAN_IO_arch as seganio
import SegAN_no_dice_arch as nodice
import DeepMRI as deepmri

dataset = {
           'training':'brats2019_training_crop_mri',
           'validation':'brats2019_validation_crop_mri',
           'testing':'brats2019_testing_crop_mri'
          }

modelname = 'Segan_NoDice_TF2_brats2019_ALL'
architecture=nodice
input_modalities = ["t1", "t1ce", "t2", "flair"]
output_labels = 1
model_input_size = 160
tracked_metric='dice_score_0'
seed=1234567890


gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
gan.load_dataset(dataset, input_modalities)
gan.build_model(seed=seed, arch=architecture)
gan.train(tracked_metric=tracked_metric)