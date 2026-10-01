# TomatoXpert
An agri-tech platform that forecasts tomato market prices using time-series modeling (Facebook Prophet) and identifies plant diseases from leaf images using deep learning (Keras, OpenCV). Stage 2 finalist in the Tomato Grand Challenge.
# TomatoXpert: Tomato Price Prediction & Leaf Disease Detection

TomatoXpert is a two-part machine learning project for tomato farming:

1. **Price forecasting:** predicts the daily modal (most common) market price of tomatoes from past prices, calendar patterns and rainfall, using an LSTM.
2. **Disease detection:** identifies 10 tomato leaf conditions (9 diseases + healthy) from photos, using a CNN.

Together they support two decisions a farmer faces: *when to sell* and *what is wrong with my crop*.

---

## 1. Tomato Price Prediction (`tomatoma.ipynb`)

### Data
- Daily tomato market prices (min, max and modal price), about 3,150 records from 2015 to 2023
- Daily rainfall data for the same period, reshaped from a monthly/day-column table into a time series and merged by date

### Approach
- **Exploratory analysis:** modal price patterns by day of week, month, weekday vs. weekend, and across years
- **Feature engineering:** day of week, month, year, rainfall, and a **7-day lag window** of past prices, calendar features and rainfall
- **Preprocessing:** missing rainfall values filled with 0, features standardized with `StandardScaler`
- **Model:** stacked LSTM (64 + 64 units) with dropout, followed by Dense(32) and Dense(1), trained with Adam and MSE loss for 50 epochs
- **Experiment:** the same model trained with and without rainfall features to measure rainfall's contribution

### Results

| Model | Test MSE |
|---|---|
| LSTM with rainfall | 153,709 |
| LSTM without rainfall | 156,389 |

Adding rainfall gave a small improvement. Prices vary a lot between days, so there is plenty of room for tuning (for example sequence-based windows or Prophet/GRU baselines).

---

## 2. Tomato Disease Detection (`tomato-disease-detection-cnn-93-5.ipynb`)

### Data
Tomato leaf image dataset with **10 classes**: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, and Healthy.

| Split | Images |
|---|---|
| Train | 7,000 |
| Validation | 3,000 |
| Test | 1,000 (100 per class) |

### Approach
- Images resized to 224x224 and rescaled to [0, 1]
- Augmentation: shear, zoom and horizontal flip
- **Model:** custom CNN with 6 Conv2D + MaxPooling blocks, then Dense(64) and a 10-way softmax
- Adam optimizer (learning rate 0.001), categorical cross-entropy, early stopping on validation accuracy

### Results
- **Test accuracy: 93.8%** on 1,000 unseen images
- Macro F1-score: 0.94
- Best classes: Mosaic virus (F1 1.00), Healthy (0.99), Yellow leaf curl virus and Spider mites (0.97)
- Hardest classes: Septoria leaf spot (recall 0.79) and Early blight (F1 0.89), which are often confused with other leaf spot diseases

---

## Tech Stack

Python, TensorFlow/Keras, OpenCV, scikit-learn, Pandas, NumPy, Matplotlib

## How to Run

```bash
pip install tensorflow opencv-python scikit-learn pandas numpy matplotlib openpyxl
```

1. **Price model:** place the price and rainfall Excel files in the project folder, update the file paths at the top of `tomatoma.ipynb`, and run all cells.
2. **Disease model:** download a tomato leaf dataset organised as `train/` and `val/` class folders, update `train_data_dir` and `test_data_dir`, and run all cells.

The datasets are not included in this repo.

## Future Work

- Add Prophet and GRU baselines for price forecasting, and report RMSE/MAE
- Use transfer learning (MobileNet, EfficientNet) to push disease accuracy higher
- Combine both models in a simple web or mobile app for farmers
