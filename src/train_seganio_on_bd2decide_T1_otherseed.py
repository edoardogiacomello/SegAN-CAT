from DeepMRI import DeepMRI
import SegAN_IO_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Segan_IO_TF2_bd2decide_on_T1_otherseed")
gan.load_dataset(dataset='bd2decide', mri_types=["T1"])
gan.build_model(load_model='last', seed=9876543210, arch=arch)
gan.train(alternating_steps=(50,50))
