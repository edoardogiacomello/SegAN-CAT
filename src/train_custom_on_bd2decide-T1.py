from DeepMRI import DeepMRI
import Custom_MultiFOV_G6_arch as arch
gan = DeepMRI(batch_size=64, size=160, mri_channels=1, model_name="Custom_MultiFOV_G6_bd2decide_on_T1_alt50")
gan.load_dataset(dataset='bd2decide', mri_types=["T1"])
gan.build_model(load_model='last', seed=1234567890, arch=arch)
gan.train(alternating_steps=(50,50))
