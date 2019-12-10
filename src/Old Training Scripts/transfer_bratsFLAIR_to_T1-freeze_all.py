from DeepMRI import DeepMRI
import SegAN_IO_arch as arch

gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Transfer_Brats_Flair_to_T1_freeze_all")
gan.build_model(load_model='models/Segan_IO_TF2_brats_on_FLAIR/best_dice_score_168-29', seed=1234567890, arch=arch)
gan.load_dataset('brats', ['MR_T1'])

# Disable training for all D layers
for l in gan.discriminator.layers:
    l.trainable = False
    
gan.train()