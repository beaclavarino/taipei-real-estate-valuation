# Taipei Real Estate Valuation

**Statistical Learning** — Master in Data Science for Economics  
*Beatrice Clavarino*

## Abstract
This report analyzes residential property values in Taipei, Taiwan, using the [Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) from the UCI Machine Learning Repository. The project investigates market segmentation and evaluates the predictive power of structural and locational characteristics on unit house prices ($\text{NTD/m}^2$).

### Key Phases:
* **Preprocessing & Feature Engineering**: Standardized spatial units ($\text{Ping} \rightarrow \text{m}^2$) and applied a logarithmic transformation ($\log(\text{Price})$) to reduce skewness and stabilize variance for regression modeling.
* **Unsupervised Learning**: Explored market structure via **Principal Component Analysis (PCA)** and **K-means clustering** ($k = 3$) on the principal components, identifying three economically meaningful segments differentiated by MRT accessibility, local amenities, and building age.
* **Supervised Learning**: Evaluated predictive accuracy on an out-of-sample test split ($70/30$) across **Linear Regression**, a **Multi-Layer Perceptron (MLP)**, and **Random Forest**.
* **Key Findings**: 
  * Proximity to MRT stations and geographic coordinates represent the primary drivers of housing value, far outweighing structural factors such as house age.
  * **Random Forest** achieved the best overall performance ($R^2 = 0.703$, $\text{RMSE} = 0.238$), confirming the prominence of non-linear spatial dependencies.

## Repository Structure
* `SLreport.pdf`: Complete academic report written in LaTeX detailing the data analysis, methodology, and results.
* `SLcode.R`: R scriptcontaining end-to-end routines for data cleaning, PCA, clustering, model training, evaluation, and visualization.
* `figures/`: High-resolution figures and diagnostic plots included in the report.
