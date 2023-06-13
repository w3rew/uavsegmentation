import segmentation_models_pytorch as smp


decoders = {
        'unet': smp.Unet,
        'unet++': smp.UnetPlusPlus
        }

def get_model(config):
    decoder = decoders[config['decoder']]

    model = decoder(**config['params'])

    return model
