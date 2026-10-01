# Data

Place the dataset here (git-ignored, not redistributed):

```
data/
├── train/
│   ├── train/   # low-resolution noisy inputs  (e.g. gt_00001.png)
│   └── gt/      # high-resolution clean targets
├── val/
│   ├── val/     # low-resolution noisy inputs
│   └── gt/      # high-resolution clean targets
└── test/        # low-resolution noisy inputs only (60 images)
```

Sizes seen: LR 160×256 → HR 640×1024 (4×). The notebook pairs files by numeric ID. Update `DATA_ROOT`/`TRAIN_DIR`/`VAL_DIR`/`TEST_DIR` in the notebook.

The competition also supplied a `submission.py` helper (image folder → CSV of pixel values) that is not redistributed here; the notebook locates and runs it.
