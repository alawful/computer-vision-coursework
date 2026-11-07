# Computer vision coursework

This repository contains my first computer-vision assignment at the University of Tehran. The notebook moves through a set of practical image-processing tasks and records the outputs used to check the implementation.

The work includes image types and color channels, contrast and lighting changes, masking, synthetic noise, filtering, edge detection, image-quality checks, and frame-based video experiments.

## Repository layout

- `notebook/` contains the submitted Jupyter notebook.
- `src/solution.py` contains the main implementation in script form.
- `outputs/` contains selected figures and small CSV/JSON validation summaries.

Large generated videos, the submitted Word report, and duplicate HTML exports are not included. This keeps the repository reviewable while preserving the code and representative evidence without publishing a student number.

## Running

Create a Python environment and install the listed dependencies:

```bash
pip install -r requirements.txt
```

The notebook expects its original input images or videos for a full rerun. The committed outputs allow the completed work to be inspected without those larger files.

## Scope

This is coursework rather than a reusable vision library. It documents my implementation and validation for the assigned tasks.
