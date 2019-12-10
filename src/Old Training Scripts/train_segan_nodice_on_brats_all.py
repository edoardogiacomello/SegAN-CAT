from DeepMRI import DeepMRI
import SegAN_no_dice_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=4, model_name="Segan_NoDice_TF2_brats_ALL")
gan.load_dataset(dataset='brats', mri_types=["MR_T1", "MR_T1c", "MR_T2", "MR_Flair"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train()
