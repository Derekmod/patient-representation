#!/bin/sh
PICKLE_DIR="/home-1/dmodzel2@jhu.edu/work/derek/PatientRepresentation"
DATA_DIR="/home-1/dmodzel2@jhu.edu/work/derek/data/V7_data"
DIMENSION=6
INERTIA=4
MAX_ITER=50
TECHNICAL_DIR="/home-1/dmodzel2@jhu.edu/work/princy/gtex/data/annotations/GTEx_Analysis_2015-01-12_Annotations_SampleAttributesDS.txt"
COVARIATE_DIR="/home-1/dmodzel2@jhu.edu/work/princy/gtex/data/annotations/GTEx_Analysis_2015-01-12_Annotations_SubjectPhenotypesDS.txt"
python evaluate_reconstruction.py "--data-dir=$DATA_DIR" --pickle-dir=$PICKLE_DIR --dimension=$DIMENSION --inertia=$INERTIA --max-iter=$MAX_ITER --technicals=$TECHNICAL_DIR --covariates=$COVARIATE_DIR
