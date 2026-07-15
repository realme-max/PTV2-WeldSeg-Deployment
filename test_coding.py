import hydra

@hydra.main(config_path='config', config_name='partseg_v2_improved')
def cout(args):
    omegaconf.OmegaConf.set_struct(args, False)
    print("hello world")