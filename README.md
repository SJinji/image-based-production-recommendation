# Fashion Product Recommendation System

This repository contains code and resources for building a fashion product recommendation system using three different methods. The dataset used is the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data), which consists of around 44,000 high-resolution images of fashion products belonging to 143 unique classes such as T-shirts, jeans, watches, etc.

## Methods Implemented

1. **Feature Extraction using ResNet50**
   - This method utilizes the pre-trained ResNet50 model to extract feature embeddings from the fashion product images. These embeddings are then used to compute similarities between products and provide recommendations.

2. **Feature Extraction Combined with Metadata**
   - In this method, we combine the image features extracted using ResNet50 with product metadata such as category, subcategory, and color. This allows for a more holistic representation of each product, leading to improved recommendation quality.

3. **CNN Classifier Based Retrieval (CCBR)**
   - A custom Convolutional Neural Network (CNN) model is trained to classify images into different product categories. Once the model is trained, the class of the input image is predicted and similar products from the same class are recommended based on the extracted feature embeddings. This method leverages both classification and retrieval techniques.

   > **Note:** Due to computational resource limitations, we have used only 10% of the data from each subCategory for training and evaluation. This reduced data size might impact the accuracy of the CCBR method. From research, it is observed that with a larger dataset, CCBR generally performs best. For more insights, you can refer to [this article on image-based product recommendation](https://zakim.medium.com/image-based-product-recommendation-e1bfa29e508).

## Dataset

The dataset is obtained from Kaggle: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data). It contains:
- **Images:** Around 44,000 high-resolution images of fashion products with a size of 2400x1600 pixels.
- **Metadata:** Each image is accompanied by metadata including the product category, subcategory, article type, and base color.

### Challenges
- **High Resolution:** The large image size poses challenges for image pre-processing and model training.
- **Class Imbalance:** The dataset contains a wide variety of classes with an imbalanced distribution.

## Results

### ResNet50 Feature Extraction
- Extracted high-quality image embeddings that capture the visual features of fashion products.
- The model can successfully recommend visually similar products based on these features.

### Metadata Combination
- Combining image features with metadata such as category and color improved the recommendation accuracy.
- This method provides a more comprehensive representation of fashion products.

### CNN Classifier Based Retrieval (CCBR)
- The CNN model achieved reasonable classification accuracy on the dataset.
- The CCBR technique enables recommendations within the same product category, enhancing user experience.
- Due to the use of a reduced dataset (10% of each subCategory), the CCBR method might not perform at its full potential. For best results, consider using the full dataset.

### Comparison of Results:
![image](https://github.com/user-attachments/assets/f233e5fa-c244-4ecf-8f1d-9f006b292ff6)
![image](https://github.com/user-attachments/assets/f32adf30-6cea-4c11-9c05-0d36873272db)

#### 1. **ResNet50 for Feature Extraction:**
   - **Visual Results:** 
     - **Top Image:** The results show a variety of items such as watches, perfume bottles, and wallets. The retrieved items are visually diverse, suggesting that the ResNet50 model is capturing general features, but not necessarily the fine-grained details related to product type.
   - **Observations:**
     - **Pros:** The method is able to retrieve visually similar items based on general shape, color, and texture.
     - **Cons:** Lack of specificity in results. The retrieval includes items from different sub-categories, indicating that ResNet50's global features may not capture detailed product-specific characteristics.

#### 2. **Combined ResNet50 with Metadata:** 
   - **Visual Results:**
     - **Bottom Image:** The results show a set of cufflinks, which are much more consistent in terms of category. This indicates that the inclusion of metadata significantly improved the focus of the recommendations.
   - **Observations:**
     - **Pros:** Better specificity in results. The retrieved items are all cufflinks, suggesting a higher relevance in recommendations due to the use of combined metadata and image features.
     - **Cons:** While the results are more focused, they may still miss subtle visual variations within the same sub-category.

The combined approach of using ResNet50 with metadata provides a more focused set of recommendations compared to using ResNet50 alone. However, the choice of method should be based on the specific use case and the desired level of recommendation granularity. For cases where detailed visual features are important within the same category, further fine-tuning or the use of more sophisticated models may be required.

## Future Work

- **Improve Classifier Accuracy:** Experiment with deeper architectures, data augmentation, and advanced training techniques to improve the CNN classifier's accuracy.
- **Integrate More Metadata:** Use additional metadata fields such as brand, season, and usage to further refine recommendations.
- **Explore Other Models:** Test other pre-trained models like EfficientNet or Vision Transformers for feature extraction.

## Co-authors
- jinji.shen@essec.edu
- mengyu.liang@essec.edu
