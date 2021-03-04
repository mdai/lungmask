import numpy as np
from skimage.measure import find_contours


def get_values_r231(outmask, ds):
    contours = find_contours(outmask, 0)
    if contours:
        outputs = []
        lcounter, rcounter = 0, 1
        for contour in contours:
            data = {"vertices": [[(v[1]), (v[0])] for v in contour.tolist()]}
            if contour[0][1] <= 250:
                output = {
                    "type": "ANNOTATION",
                    "study_uid": str(ds.StudyInstanceUID),
                    "series_uid": str(ds.SeriesInstanceUID),
                    "instance_uid": str(ds.SOPInstanceUID),
                    "class_index": rcounter,
                    "data": data,
                }
                rcounter = 2
            else:
                output = {
                    "type": "ANNOTATION",
                    "study_uid": str(ds.StudyInstanceUID),
                    "series_uid": str(ds.SeriesInstanceUID),
                    "instance_uid": str(ds.SOPInstanceUID),
                    "class_index": lcounter,
                    "data": data,
                }
                lcounter = 2
            outputs.append(output)
    else:
        outputs = [
            {
                "type": "NONE",
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "instance_uid": str(ds.SOPInstanceUID),
            }
        ]
    return outputs


def get_values_ltrclobes(masks, ds):
    if masks:
        preds = []
        for submask, label in masks:
            output = {
                "type": "ANNOTATION",
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "instance_uid": str(ds.SOPInstanceUID),
                "class_index": int(label),
                "data": {"mask": submask.tolist()},
            }
            preds.append(output)
    else:
        preds = [
            {
                "type": "NONE",
                "study_uid": str(ds.StudyInstanceUID),
                "series_uid": str(ds.SeriesInstanceUID),
                "instance_uid": str(ds.SOPInstanceUID),
            }
        ]
    return preds
