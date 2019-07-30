from DeepMRI import DeepMRI
import SegAN_IO_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Segan_IO_TF2_brats_on_T1c")
gan.load_dataset(dataset='brats', mri_types=["MR_T1c"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train()