import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

app = Flask(__name__)
# from app import prediction
df = pd.read_csv("weatherHistory.csv")

# Preprocessing
# Drop NaN Value
df = df.dropna()
# Drop record contains 'and' substring in target column (Summary)
andIdx = df[df['Summary'].str.contains("and")].index
df.drop(andIdx , inplace=True)
# encoding
label_encoder = preprocessing.LabelEncoder()
df['Precip Type'] = label_encoder.fit_transform(df['Precip Type'])
df['Summary'] = label_encoder.fit_transform(df['Summary'])
# fill NA with median
med = df['Precip Type'].median()
df['Precip Type'].fillna(med, inplace=True)

# Prepare Data Train & Data Test
features = ['Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
            'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
target = 'Summary'

# data training 70%
# data testing 30%
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target],
                                                    train_size=0.85, test_size=0.15, shuffle=False)
# print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
# print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

'''
# Train Model Decision Tree
model = DecisionTreeClassifier(max_depth=18)
model.fit(x_train, y_train)
'''

# Train Model Bagging Classifier
dtc = DecisionTreeClassifier(criterion="entropy")
bag_model = BaggingClassifier(base_estimator=dtc, n_estimators=100, bootstrap=True)
bag_model = bag_model.fit(x_train, y_train)

'''
# Accuracy Testing
predicted = bag_model.predict(x_test)
print(bag_model.score(x_test, y_test))

# Confusion Matrix
print(confusion_matrix(y_test, predicted))

# Measure Performance
print(classification_report(y_test, predicted))
'''

# tampilan index (halaman awal)
@app.route('/')
def index():
    # menampilkan template
    return render_template('index.html', result="?", precipType = "?", temperature = "?",
    apparentTemp = "?", humidity = "?", windSpeed = "?", windBearing = "?", visibility = "?",
    pressure = "?")

@app.route('/prediction', methods=['POST'])
def prediction():
  # Classification Result
  precipType = int(request.form['precipType'])
  temperature = float(request.form['temperature'])
  apparentTemp = float(request.form['apparentTemp'])
  humidity = float(request.form['humidity'])
  windSpeed = float(request.form['windSpeed'])
  windBearing = float(request.form['windBearing'])
  visibility = float(request.form['visibility'])
  pressure = float(request.form['pressure'])

  predicted = bag_model.predict([[precipType, temperature, apparentTemp, humidity, windSpeed, windBearing, visibility, pressure]])
  result = ""

  precipType = "Snow" if precipType == 1 else "Rain"

  if predicted[0] == 0:
    result = "Breezy"
  elif predicted[0] == 1:
    result = "Clear"
  elif predicted[0] == 2:
    result = "Drizzle"
  elif predicted[0] == 3:
    result = "Dry"
  elif predicted[0] == 4:
    result = "Foggy"
  elif predicted[0] == 5:
    result = "Light Rain"
  elif predicted[0] == 6:
    result = "Mostly Cloudy"
  elif predicted[0] == 7:
    result = "Overcast"
  elif predicted[0] == 8:
    result = "Partly Cloudy"
  elif predicted[0] == 9:
    result = "Rain"
  elif predicted[0] == 10:
    result = "Windy"
  else:
    result = "Not Found"
  print("Prediction Result :", result)

  return render_template('index.html', result = result, precipType = precipType, temperature = temperature,
    apparentTemp = apparentTemp, humidity = humidity, windSpeed = windSpeed, windBearing = windBearing,
    visibility = visibility, pressure = pressure)

# drive
if __name__ == '__main__':
    app.run(debug=True)