import os
import numpy as np
import torch
import warnings
import sys
import logging

from resunet import UNet
from utils import preprocess, postrocessing, reshape_mask

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# stores urls and number of classes of the models
model_urls = {
    ('unet', 'r231'): ('unet_r231-d5d2fc3d.pth', 3),
    ('unet', 'ltrclobes'): ('unet_ltrclobes-3a07043d.pth', 6),
    ('unet', 'r231covidweb'): ('unet_r231covid-0de78a7e.pth', 3)
}

def apply(image, model, device, volume_postprocessing=True):
    tvolslices, xnew_box = preprocess(image, resolution=[256, 256])
    tvolslices[tvolslices > 600] = 600
    tvolslices = np.divide((tvolslices + 1024), 1624)
    timage_res = np.empty((np.append(0, tvolslices[0].shape)), dtype=np.uint8)

    with torch.no_grad():
        X = torch.Tensor(tvolslices).unsqueeze(0).to(device)
        prediction = model(X)
        pls = torch.max(prediction, 1)[1].detach().cpu().numpy().astype(np.uint8)
        timage_res = np.vstack((timage_res, pls))

    # postprocessing includes removal of small connected components, hole filling and mapping of small components to
    # neighbors
    if volume_postprocessing:
        outmask = postrocessing(timage_res)
    else:
        outmask = timage_res

    outmask = np.asarray(
        [reshape_mask(outmask[i], xnew_box[i], image.shape[1:]) for i in range(outmask.shape[0])],
        dtype=np.uint8)

    return outmask.astype(np.uint8)


def get_model(modeltype, modelname, modelpath, device):
    model_url, n_classes = model_urls[(modeltype, modelname)]
    model_file = os.path.join(modelpath, "../model", model_url)
    if device.type == "cpu":
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(model_file)

    if modeltype == 'unet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
    elif modeltype == 'resunet':
        model = UNet(n_classes=n_classes, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=True)
    else:
        logging.exception(f"Model {modelname} not known")
    model.load_state_dict(state_dict)
    model.eval()
    return model

'''
def apply_fused(image, basemodel = 'LTRCLobes', fillmodel = 'R231', volume_postprocessing=True):
    # Will apply basemodel and use fillmodel to mitiage false negatives
    mdl_r = get_model('unet',fillmodel)
    mdl_l = get_model('unet',basemodel)
    logging.info("Apply: %s" % basemodel)
    res_l = apply(image, mdl_l, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    logging.info("Apply: %s" % fillmodel)
    res_r = apply(image, mdl_r, force_cpu=force_cpu, batch_size=batch_size,  volume_postprocessing=volume_postprocessing, noHU=noHU)
    spare_value = res_l.max()+1
    res_l[np.logical_and(res_l==0, res_r>0)] = spare_value
    res_l[res_r==0] = 0
    logging.info("Fusing results... this may take up to several minutes!")
    return utils.postrocessing(res_l, spare=[spare_value])
'''