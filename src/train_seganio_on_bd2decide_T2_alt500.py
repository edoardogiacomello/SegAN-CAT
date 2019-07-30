from DeepMRI import DeepMRI
import SegAN_IO_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Segan_IO_TF2_bd2decide_on_T2_alt500")
gan.load_dataset(dataset='bd2decide', mri_types=["T2"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train(alternating_steps=(500,500))
