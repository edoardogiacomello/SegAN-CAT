from DeepMRI_GAP import DeepMRI
import SegAN_IO_GAP_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=4, model_name="Segan_IO_GAP_TF2_brats_ALL_alt50")
gan.load_dataset(dataset='brats', mri_types=["MR_T1", "MR_T1c", "MR_T2", "MR_Flair"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train(alternating_steps=(50,50))
