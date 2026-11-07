# Mice Protein Expression: Clustering Before and After PCA

A mini project for our unsupervised learning course. We run K-Means on the data first with all the proteins, then again after PCA reduces them to 2 components, and compare the results.

**Team:** Luluh, Aleen, Reema, Khalid

**Live demo:** [mice-protein-expression.streamlit.app](https://mice-protein-expression.streamlit.app/)

## Dataset

[Mice Protein Expression (UCI)](https://archive.ics.uci.edu/dataset/342/mice+protein+expression)

- 1,080 samples from 72 mice (15 measurements per mouse)
- 77 protein levels per sample
- 8 classes, from genotype, treatment and behavior
- The labels are never used for clustering, only to check the results

## Steps

1. **Preprocessing:** check for duplicates (none), check missing values (1.68%), fill them with the median, and scale with StandardScaler
2. **K-Means before PCA:** on all 77 proteins
3. **PCA:** reduce the data to 2 components
4. **K-Means after PCA:** on the 2 components
5. **Comparison:** silhouette scores and 2D plots
6. **Bonus:** try different numbers of components, and hierarchical clustering

## Results (k = 8)

| | Silhouette score |
|---|---|
| Before PCA | 0.123 |
| After PCA | 0.345 |

- PCA with 2 components keeps 42.3% of the variance
- The score is higher after PCA and the plot is clearer, but we lose information
- The higher score is partly because 2D data is easier to separate

## Files

- `PCA_Mice_Protein (1).ipynb`: the notebook with all the steps
- `Data_Cortex_Nuclear.xls`: the dataset
- `app.py`: the Streamlit app (Pinky and the Brain theme)
- `components/`: the app's charts and animations
- `requirements.txt`: the packages needed

## Run the app

Open the [live demo](https://mice-protein-expression.streamlit.app/), or run it on your computer:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, if `streamlit` is not recognized, use `py -m streamlit run app.py`.
