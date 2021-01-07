import os
from io import BytesIO
import numpy as np
import torch
import pydicom

from mask import get_model, apply
from helper import get_values_r231, get_values_ltrclobes

# Currently supported model names - R231, LTRCLobes, R231CovidWeb
args = {
    "model_type": "unet",
    "model_name": "LTRCLobes",
    "postprocess": True,
}


class MDAIModel:
    def __init__(self):
        self.model_type = args.get("model_type", "unet").lower()
        self.model_name = args.get("model_name").lower()
        self.postprocessing = args.get("postprocess", True)

        root_path = os.path.dirname(os.path.dirname(__file__))

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_ids = list(range(torch.cuda.device_count()))
        else:
            self.device = torch.device("cpu")
            gpu_ids = []

        self.model = get_model(self.model_type, self.model_name, root_path, self.device)
        self.model.to(self.device)

    def predict(self, data):
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []

        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            ds = pydicom.dcmread(BytesIO(file["content"]))
            image = ds.pixel_array

            if image.dtype == "int16":
                image = image.astype("int32") - 1024

            if len(image.shape) == 2:
                image = np.expand_dims(image, 0)

            outmask = apply(
                image,
                self.model,
                self.device,
                volume_postprocessing=self.postprocessing,
            )
            outmask = outmask.squeeze(0)

            if self.model_name == "ltrclobes":
                vals = set(np.unique(outmask))
                masks = [
                    (np.uint8(outmask) == i, i - 1) for i in range(1, 6) if i in vals
                ]
                outputs += get_values_ltrclobes(masks, ds)
            else:
                outputs += get_values_r231(outmask, ds)
        return outputs
