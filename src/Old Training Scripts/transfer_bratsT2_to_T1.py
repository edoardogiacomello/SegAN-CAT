from DeepMRI import DeepMRI
import SegAN_IO_arch as arch

gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Transfer_Brats_T2_to_T1")
gan.build_model(load_model='models/Segan_IO_TF2_brats_on_T2/best_dice_score_182-25', seed=1234567890, arch=arch)
gan.load_dataset('brats', ['MR_T1'])

gan.train()