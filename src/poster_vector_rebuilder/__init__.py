__version__ = "0.1.0"

# OpenCV has returned HoughLinesP results as either (N, 1, 4) or (N, 4)
# across builds/versions. The reconstruction code uses the documented
# (N, 1, 4)-style shape, so normalize the package-local runtime consistently.
import cv2 as _cv2
import numpy as _np

if not getattr(_cv2.HoughLinesP, "_pvr_shape_compat", False):
    _original_hough_lines_p = _cv2.HoughLinesP

    def _hough_lines_p_compat(*args, **kwargs):
        result = _original_hough_lines_p(*args, **kwargs)
        if result is None:
            return None
        array = _np.asarray(result)
        if array.ndim == 2 and array.shape[-1] == 4:
            array = array[:, None, :]
        return array

    _hough_lines_p_compat._pvr_shape_compat = True
    _cv2.HoughLinesP = _hough_lines_p_compat
