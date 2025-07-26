from flask import Flask, render_template, request
import pickle
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load trained model and encoders
with open("model.pkl", "rb") as f:
    model, label_encoders, target_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Capture form inputs
            input_data = {
                "Country Code": int(request.form["country_code"]),
                "City": request.form["city"],
                "Average Cost for two": float(request.form["cost"]),
                "Price range": int(request.form["price_range"]),
                "Has Table booking": request.form["table_booking"],
                "Has Online delivery": request.form["online_delivery"],
                "Aggregate rating": float(request.form["rating"]),
                "Votes": int(request.form["votes"])
            }

            # Convert to DataFrame
            df = pd.DataFrame([input_data])

            # Apply encoders
            for col in ["City", "Has Table booking", "Has Online delivery"]:
                if col in label_encoders:
                    df[col] = label_encoders[col].transform(df[col].astype(str))

            # Predict
            cuisine_pred = model.predict(df)[0]
            prediction = target_encoder.inverse_transform([cuisine_pred])[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
