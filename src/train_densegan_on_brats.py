from DeepMRI import DeepMRI
import DenseGAN_arch as arch
gan = DeepMRI(batch_size=32, size=160, mri_channels=2, model_name='DenseGan_brats')
gan.load_dataset('brats')
gan.build_model(load_model=True, seed=1234567890, arch=arch)
gan.train()
