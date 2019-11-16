import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.float32):
            return obj.tolist()

        return json.JSONEncoder.default(self, obj)
