from DeepMRI import DeepMRI
import Custom_MultiFOV_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Custom_MultiFOV_brats_on_FLAIR")
gan.load_dataset(dataset='brats', mri_types=["MR_Flair"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train(alternating_steps=None)
