import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] 
sub_modules = ['', 'object_detection', 'text_line_detection', 'text_line_recognition', 'kie', 'postprocess']
for m in sub_modules:
    path = os.path.join(str(ROOT), m)
    print(path)
    if path not in sys.path:
        sys.path.append(path)

from .predictor import Predictor