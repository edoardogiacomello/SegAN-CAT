#import SegAN_arch as seganorig
import SegAN_IO_arch as seganio
import DeepMRI as deepmri

dataset = {
           'training':'brats2019_training_crop_mri',
           'validation':'brats2019_validation_crop_mri',
           'testing':'brats2019_testing_crop_mri'
          }

base_checkpoint = 'models/Segan_IO_TF2_brats2019_on_FLAIR/best_dice_score_0_89-10'
MAX_EPOCHS = 300

# Freeze discriminator
for target_modality in ["t1", "t1ce", "t2"]:
    output_labels = 1
    model_input_size = 160
    tracked_metric='dice_score_0'
    seed=1234567890
    input_modalities = [target_modality]
    modelname = 'Transfer_Brats2019_Flair_to_{}_freeze_all'.format(target_modality)
    architecture=seganio
    gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
    gan.build_model(load_model=base_checkpoint, transfer=True, seed=seed, arch=architecture)
    gan.load_dataset(dataset, input_modalities)
    
    # Disable training for all D layers
    for l in gan.discriminator.layers:
        l.trainable = False
        
    gan.train(tracked_metric=tracked_metric, max_epochs=MAX_EPOCHS)
    del gan

# Full training
for target_modality in ["t1", "t1ce", "t2"]:
    output_labels = 1
    model_input_size = 160
    tracked_metric='dice_score_0'
    seed=1234567890
    input_modalities = [target_modality]
    modelname = 'Transfer_Brats2019_Flair_to_{}'.format(target_modality)
    architecture=seganio
    gan = deepmri.DeepMRI(batch_size=64, size=model_input_size, mri_channels=len(input_modalities), output_labels=output_labels, model_name=modelname)
    gan.build_model(load_model=base_checkpoint, transfer=True, seed=seed, arch=architecture)
    gan.load_dataset(dataset, input_modalities)
    
    gan.train(tracked_metric=tracked_metric, max_epochs=MAX_EPOCHS)
    del gan






