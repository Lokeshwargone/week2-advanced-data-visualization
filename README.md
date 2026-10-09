# Week 2 — Advanced Data Visualization and Storytelling with Python

## Contents
- `Week_2_Advanced_Data_Visualization_and_Storytelling.docx`: main report with six annotated visualizations.
- `create_visual_story.py`: reproducible script that generates the charts.
- `figures/`: generated chart images.
- `diabetes_dataset_with_target.csv`: dataset loaded from scikit-learn and saved with the target column.

## Dataset
The report uses the publicly documented Diabetes dataset included with scikit-learn (`sklearn.datasets.load_diabetes`). It contains 442 observations, 10 baseline features, and a one-year disease progression target. The feature values are standardized by the dataset provider; they are not original clinical units.

## Reproduce
Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Then run:

```bash
python create_visual_story.py
```

The script creates six PNG visualizations in `figures/` and exports the data to `diabetes_dataset_with_target.csv`.

## Important interpretation note
The analysis is exploratory. Correlation is not causation, and the charts are not medical advice, a diagnostic tool, or a clinically validated prediction system.
