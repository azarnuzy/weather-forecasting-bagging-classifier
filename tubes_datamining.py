import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

app = Flask(__name__)
# from app import prediction
df = pd.read_csv("weatherHistory.csv")

#drop data dalam kolom yang masih berisi nAn
df = df.dropna()

#drop data dalam kolom summary yang berisi kata 'and'
andIdx = df[df['Summary'].str.contains("and")].index
df.drop(andIdx , inplace=True)

# #melihat data unik dalam summary
# df['Summary'].unique()

# df.columns

# Preprocessing
label_encoder = preprocessing.LabelEncoder()
df['Precip Type'] = label_encoder.fit_transform(df['Precip Type'])
df['Summary'] = label_encoder.fit_transform(df['Summary'])

# df['Precip Type'].isnull().sum()
med = df['Precip Type'].median()
df['Precip Type'].fillna(med, inplace=True)

# df['Summary'].unique()

# Prepare Data Train & Data Test
features = ['Precip Type', 'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
            'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']
target = 'Summary'
# data training 85%
# data testing 15%
x_train, x_test, y_train, y_test = train_test_split(df[features], df[target],
                                                    train_size=0.85, test_size=0.15, shuffle=False)
# print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
# print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

# Train Model
model = DecisionTreeClassifier(max_depth=18)
model.fit(x_train, y_train)

# # Testing Model
# print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
# predicted = model.predict(x_test)
# print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))

# # Measure Performance (Cross Validation)
# print("Cross Validation Accuracy: \n")
# cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
# print("Accuracy using 10 folds: ")
# print(cv_accuracy)

# # Measure Performance (Confusion Matrix)
# print("Mean accuracy: {}".format(cv_accuracy.mean()))
# print("Standard Deviation: {}".format(cv_accuracy.std()))

# # Measure Performance
# print('Precision, Recall and f-1 Scores for Decision Tree Classifier\n')
# print(classification_report(y_test, predicted))

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

  predicted = model.predict([[precipType, temperature, apparentTemp, humidity, windSpeed, windBearing, visibility, pressure]])
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