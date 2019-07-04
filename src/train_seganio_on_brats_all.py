from DeepMRI import DeepMRI
import SegAN_IO_arch as arch
gan = DeepMRI(batch_size=32, size=160, mri_channels=4, model_name="Segan_IO_TF2_brats")
gan.load_dataset('brats')
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train()
