This folder contains the data-preparation code sample for the project:

I chose this sample because it is the most data-oriented part of my waste
classification project from spring quarter. 

# What The Code Shows:

- Data cleaning and transformation from COCO-style JSON annotations into
  YOLO-formatted training labels.
- Reproducible sample splitting using an explicit random seed.
- Modular helper functions that can be unit-tested independently.
- Clear command-line arguments and concise documentation.
- Defensive checks for missing annotation files, invalid split fractions, and
  out-of-bounds bounding boxes.

# How To Run:

From the project root, after downloading the TACO dataset, run the following in bash:

python code_sample/prepare_taco_yolo_dataset.py \
    --taco-dir training/datasets/TACO \
    --out training/datasets/taco_yolo \
    --val-split 0.2 \
    --seed 42


The script writes:

`images/train/` and `images/val/`
`labels/train/` and `labels/val/`
`data.yaml`, which can be passed to a YOLO training job

# Additional Coursework Context:

Campus Waste Sorting Assistant is my open-ended project. I also have classwork
from CS349 that covers related machine-learning and data-analysis foundations,
including decision trees, logistic regression, polynomial regression, Naive
Bayes, sparse feature matrices, model evaluation, and fairness metrics. I chose
this project sample because it is more self-contained and project-oriented,
while the CS349 work shows additional practice implementing statistical and
machine-learning methods from scratch.
