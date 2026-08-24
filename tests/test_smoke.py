"""Smoke tests: verify all modules parse, import and key functions behave.

Run with: pytest tests/ -v
These tests do NOT require model weights or GPU.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def test_all_scripts_compile():
    """Every top-level script must be valid Python 3."""
    scripts = [
        'run.py', 'webapp.py', 'vehicle-detection.py',
        'license-plate-detection.py', 'license-plate-ocr.py',
        'gen-outputs.py', 'create-model.py', 'train-detector.py',
        'annotation-tool.py',
    ]
    for name in scripts:
        path = PROJECT_ROOT / name
        assert path.exists(), '%s missing' % name
        compile(path.read_text(), name, 'exec')


def test_src_modules_import():
    from src.label import Label, Shape, readShapes, writeShapes, lread, lwrite
    from src.utils import IOU, IOU_centre_and_dims, nms, im2single, getWH
    from src.projection_utils import find_T_matrix, getRectPts, perspective_transform
    assert callable(IOU) and callable(nms)


def test_label_roundtrip():
    import numpy as np
    from src.label import Label, lwrite, lread

    lab = Label(cl=1, tl=np.array([0.1, 0.2]), br=np.array([0.3, 0.4]), prob=0.9)
    path = '/tmp/_test_label.txt'
    lwrite(path, [lab])
    loaded = lread(path)
    assert len(loaded) == 1
    assert np.allclose(loaded[0].tl(), [0.1, 0.2])
    assert np.allclose(loaded[0].br(), [0.3, 0.4])
    assert abs(loaded[0].prob() - 0.9) < 1e-5


def test_iou():
    from src.utils import IOU
    # identical boxes -> 1.0
    v = IOU([0, 0], [1, 1], [0, 0], [1, 1])
    assert abs(v - 1.0) < 1e-6
    # disjoint boxes -> 0.0
    v = IOU([0, 0], [1, 1], [2, 2], [3, 3])
    assert abs(v - 0.0) < 1e-6


def test_nms():
    import numpy as np
    from src.label import Label
    from src.utils import nms

    a = Label(0, np.array([0., 0.]), np.array([1., 1.]), prob=0.9)
    b = Label(0, np.array([0.05, 0.05]), np.array([1.05, 1.05]), prob=0.8)  # overlaps a
    c = Label(0, np.array([5., 5.]), np.array([6., 6.]), prob=0.7)          # separate
    picked = nms([a, b, c], iou_threshold=.5)
    assert len(picked) == 2


def test_webapp_routes_exist():
    import webapp  # noqa: F401 — imports must succeed without models loaded
    rules = {r.rule for r in webapp.app.url_map.iter_rules()}
    assert '/' in rules
    assert '/health' in rules


def test_pyproject_parse():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    data = tomllib.loads((PROJECT_ROOT / 'pyproject.toml').read_text())
    assert 'dependencies' in data['project']
