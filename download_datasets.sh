#!/bin/bash
cd datasets

# Ecommerce Customers Dataset
curl -L -o ./ecommerce-customers.zip\
  https://www.kaggle.com/api/v1/datasets/download/srolka/ecommerce-customers
unzip ./ecommerce-customers.zip

# Position Salaries Dataset
curl -L -o ./position-salariescsv.zip\
  https://www.kaggle.com/api/v1/datasets/download/mariospirito/position-salariescsv
unzip ./position-salariescsv.zip

# Mushrooms Dataset
#!/bin/bash
curl -L -o ./mushroom-classification.zip\
  https://www.kaggle.com/api/v1/datasets/download/uciml/mushroom-classification
unzip ./mushroom-classification.zip

# IBM home value dataset
curl -L https://raw.githubusercontent.com/IBM/ml-learning-path-assets/master/data/predict_home_value.csv -o ./predict_home_value.csv

rm -rf *.zip
