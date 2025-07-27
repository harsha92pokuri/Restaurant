from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect input values from form
            user_input = {
                'Votes': int(request.form['Votes']),
                'Restaurant ID': float(request.form['Restaurant_ID']),
                'Is delivering now': request.form['Is_delivering_now'],
                'Price range': int(request.form['Price_range']),
                'Has Online delivery': request.form['Has_Online_delivery'],
                'Has Table booking': request.form['Has_Table_booking']
            }

            # Create a DataFrame
            df_input = pd.DataFrame([user_input])

            # Encode categorical values
            for col in ['Is delivering now', 'Has Online delivery', 'Has Table booking']:
                df_input[col] = encoders[col].transform(df_input[col].astype(str))

            # Predict rating
            prediction = model.predict(df_input)[0]
            prediction = round(prediction, 2)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

