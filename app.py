from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

df=pd.read_csv('Fish.csv')

# Encode categorical data
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Train RandomForestClassifier model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(df[['Weight', 'Length1']], df['Species'])

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = float(request.form['weight'])
    length1 = float(request.form['length1'])

    input_features = [[weight, length1]]
    predicted_species = clf.predict(input_features)
    predicted_species = le.inverse_transform(predicted_species)[0]

    return render_template('result.html', predicted_species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
