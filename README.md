# Taipei Real Estate Valuation
## Abstract
This report analyzes residential property values in Taipei, Taiwan, using the
*Real Estate Valuation dataset* from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set).

The study aims to identify market segments and evaluate the predictive power of structural and locational features on house prices.

### Key Phases:

**Unsupervised Learning**: Explored market structure via **Principal Component Analysis (PCA)** and **K-means clustering**, identifying three distinct property groups based on MRT accessibility, amenities, and building age.

**Supervised Learning**: Compared multiple models to predict house prices, including **Linear Regression**, **Neural Networks (MLP)**, and **Random Forest**.

**Best Performer**: The Random Forest model achieved the highest accuracy ($R^2 = 0.703$), highlighting the importance of non-linear spatial relationships.
