import base64
from dotenv import load_dotenv
import os
from flask import Flask, jsonify, make_response, request
from functools import wraps
import jwt
import io
import os
import sys
import setuptools
import tokenize
import pandas as pd
import numpy as np
from pandas import json_normalize
from PIL import Image
import pytesseract
from sktime.forecasting.fbprophet import Prophet
import pytesseract
import random
import sys

app = Flask(__name__)

# Load Environment Variables
load_dotenv()
JWT_SECRET = os.environ.get("JWT_SECRET")

# Functions


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"]

        if not token:
            return make_response({"success": False, "message": "Token not found"}, 201)

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_iat":False})
        except Exception as e:
            print(e)
            return make_response(
                {"success": False, "message": "Error while decoding JWT", "error": "${}".format(e)}, 201
            )

        return f(data, *args, **kwargs)

    return decorator


# Main Routes


@app.route("/")
def index():
    return make_response(
        {"success": True, "message": "Welcome to Invoice Flask Server"}
    )


@app.route("/validate-upload", methods=["POST"])
@token_required
def validateUpload(_):
    requestBody = request.json
    base64Image = requestBody["image"]
    companyName = str(requestBody["companyName"])

    imgstring = base64Image.split("base64,")[-1].strip()

    image_string = io.BytesIO(base64.b64decode(imgstring))
    image = Image.open(image_string)

    # pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract\tesseract.exe'

    extractedInformation = str(pytesseract.image_to_string(image))

    checkString = extractedInformation.lower()
    companyNameString = companyName.lower()

    deedIndex = checkString.find("deed")
    partnershipIndex = checkString.find("partnership")
    companyNameIndex = checkString.find(companyNameString)

    if companyNameIndex == -1:
        return make_response(
            {
                "success": True,
                "message": "Image File Validated",
                "information": extractedInformation,
            }
        )
    else:
        return make_response(
            {
                "success": False,
                "message": "Image File Invalid",
                "information": extractedInformation,
            }
        )


def generate_random_float(start, end):
    return random.uniform(start, end)


def process_dataframe(df, min_value):
    for index, row in df.iterrows():
        lower = row["Lower"]
        upper = row["Upper"]

        if lower < 0 and upper < 0:
            df.at[index, "Lower"] = 0
            df.at[index, "Upper"] = 0
        elif lower < 0 and upper >= 0:
            if upper > min_value:
                df.at[index, "Lower"] = min_value * generate_random_float(0.8, 2.6)
            else:
                df.at[index, "Lower"] = min_value * generate_random_float(0.4, 0.75)
                df.at[index, "Upper"] = min_value * generate_random_float(1.25, 1.93)

    return df


def make_forecast(data):
    try:

        sys.setrecursionlimit(5000)
        df = json_normalize(data)

        # Drop Irrelevant Index Column
        df = df.drop(["index"], axis=1)

        # Get a list of all the columns
        column_list = df.columns

        # Create a list of exogenous variables
        exogenous_column_list = []

        try:
            try:
                for x in column_list:
                    if x == "date" or x == "pred":
                        continue
                    else:
                        exogenous_column_list.append(x)
            except Exception as e:
                print(e)
                return make_response(
                   {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while appending to exogenous" }
                )
                

            if len(exogenous_column_list) > 0:
                df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
                min = df["pred"].mean() / (len(df["pred"]))
                df = df.set_index("date")

                x = df.loc[:, exogenous_column_list]
                df = df.drop(exogenous_column_list, axis=1)

                df = df.squeeze(axis=1)
                x = x.squeeze(axis=1)

                df.index = pd.DatetimeIndex(df.index).to_period("M")
                x.index = pd.DatetimeIndex(x.index).to_period("M")

                fh = np.arange(5) + 1

                # Train the Prophet model with exogenous data
                exogenousForecaster = Prophet(
                    add_country_holidays={"country_name": "India"},
                    seasonality_mode="additive",
                    n_changepoints=4,
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    add_seasonality={
                        "name": "monthly",
                        "period": 30.5,
                        "fourier_order": 5,
                        "mode": "additive",
                    },
                )

                forecaster = Prophet(
                    add_country_holidays={"country_name": "India"},
                    seasonality_mode="multiplicative",
                    n_changepoints=4,
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    add_seasonality={
                        "name": "monthly",
                        "period": 30.5,
                        "fourier_order": 5,
                        "mode": "multiplicative",
                    },
                )

                # forecaster = Prophet(mcmc_samples=1200, add_country_holidays={"country_name": "India"}, seasonality_mode='additive', n_changepoints=4, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, add_seasonality={"name": 'monthly', "period": 30.5, "fourier_order": 5, "mode": "additive"})

                try:
                    exogenousModel = exogenousForecaster.fit(x)
                    model = forecaster.fit(df, x)
                    # model = forecaster.fit(y_train)

                    exogenousPredictions = exogenousModel.predict(fh=fh)

                    predictions = model.predict_interval(
                        fh=fh, X=exogenousPredictions, coverage=0.9
                    )

                    predictions.columns = predictions.columns.to_flat_index()
                    predictions = predictions.reset_index()
                    predictions.columns = ["Date", "Lower", "Upper"]

                    predictions = process_dataframe(predictions, min)

                    json_data = predictions.to_json(orient="records")

                    return json_data


                except Exception as e:
                    print(e)
                    return make_response(
                        {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while generating predictions" }
                    )
                

               
            else:
                df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
                df = df.set_index("date")
                min = df["pred"].mean() / (len(df["pred"]))

                df = df.drop(exogenous_column_list, axis=1)

                df = df.squeeze(axis=1)

                df.index = pd.DatetimeIndex(df.index).to_period("M")

                fh = np.arange(5) + 1

                # Train the Prophet model with exogenous data
                forecaster = Prophet(
                    add_country_holidays={"country_name": "India"},
                    seasonality_mode="multiplicative",
                    n_changepoints=4,
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    add_seasonality={
                        "name": "monthly",
                        "period": 30.5,
                        "fourier_order": 5,
                        "mode": "multiplicative",
                    },
                )

                # forecaster = Prophet(mcmc_samples=1200, add_country_holidays={"country_name": "India"}, seasonality_mode='additive', n_changepoints=4, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, add_seasonality={"name": 'monthly', "period": 30.5, "fourier_order": 5, "mode": "additive"})

                model = forecaster.fit(df)
                # model = forecaster.fit(y_train)

                predictions = model.predict_interval(fh=fh, coverage=0.9)

                predictions.columns = predictions.columns.to_flat_index()
                predictions = predictions.reset_index()
                predictions.columns = ["Date", "Lower", "Upper"]

                predictions = process_dataframe(predictions, min)

                json_data = predictions.to_json(orient="records")

                return json_data
        except Exception as e:
            print(e)
            return make_response(
                {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while make_forecast" }
            )
    except Exception as e:
        print(e)
        return make_response(
            {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while make_forecast" }
        )


def make_short_forecast(data):
    df = json_normalize(data)

    # Drop Irrelevant Index Column
    df = df.drop(["index"], axis=1)

    # Get a list of all the columns
    column_list = df.columns

    # Create a list of exogenous variables
    exogenous_column_list = []

    for x in column_list:
        if x == "date" or x == "pred":
            continue
        else:
            exogenous_column_list.append(x)

    if len(exogenous_column_list) > 0:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        min = df["pred"].mean() / (len(df["pred"]))
        df = df.set_index("date")

        x = df.loc[:, exogenous_column_list]
        df = df.drop(exogenous_column_list, axis=1)

        df = df.squeeze(axis=1)
        x = x.squeeze(axis=1)

        df.index = pd.DatetimeIndex(df.index).to_period("M")
        x.index = pd.DatetimeIndex(x.index).to_period("M")

        fh = np.arange(5) + 1

        # Train the Prophet model with exogenous data
        exogenousForecaster = Prophet(
            add_country_holidays={"country_name": "India"},
            seasonality_mode="multiplicative",
            n_changepoints=4,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        forecaster = Prophet(
            add_country_holidays={"country_name": "India"},
            seasonality_mode="multiplicative",
            n_changepoints=4,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        exogenousModel = exogenousForecaster.fit(x)
        model = forecaster.fit(df, x)

        exogenousPredictions = exogenousModel.predict(fh=fh)

        predictions = model.predict_interval(
            fh=fh, X=exogenousPredictions, coverage=0.9
        )

        predictions.columns = predictions.columns.to_flat_index()
        predictions = predictions.reset_index()
        predictions.columns = ["Date", "Lower", "Upper"]

        predictions = process_dataframe(predictions, min)

        json_data = predictions.to_json(orient="records")

        return json_data

    else:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
        min = df["pred"].mean() / (len(df["pred"]))
        df = df.set_index("date")

        df = df.drop(exogenous_column_list, axis=1)

        df = df.squeeze(axis=1)

        df.index = pd.DatetimeIndex(df.index).to_period("M")

        fh = np.arange(5) + 1

        # Train the Prophet model with exogenous data
        forecaster = Prophet(
            add_country_holidays={"country_name": "India"},
            seasonality_mode="multiplicative",
            n_changepoints=4,
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
        )

        model = forecaster.fit(df)

        predictions = model.predict_interval(fh=fh, coverage=0.9)

        predictions.columns = predictions.columns.to_flat_index()
        predictions = predictions.reset_index()
        predictions.columns = ["Date", "Lower", "Upper"]

        predictions = process_dataframe(predictions, min)

        json_data = predictions.to_json(orient="records")

        return json_data


@app.route("/get-forecast", methods=["POST"])
@token_required
def makeForecast(data):
    try:
        data = request.json

        monthData = None

        if "monthData" in data == False:
            return make_response(
                {"success": False, "message": "Please provide month data"}
            )
        else:
            monthData = data["monthData"]

        if len(monthData) <= 12:
            forecast_data = make_short_forecast(monthData)
            return make_response(
                {
                    "success": True,
                    "message": "Made Forecasts successfully",
                    "forecast": forecast_data,
                }
            )
        else:
            try:
                forecast_data = make_forecast(monthData)
                try:
                    return make_response(
                        {
                            "success": True,
                            "message": "Made Forecasts successfully",
                            "forecast": forecast_data,
                        }   
                    )
                except Exception as e:
                    print(e)
                    return make_response(
                        {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while generating forecast for the response data" }
                    )

            except Exception as e:
                print(e)
                return make_response(
                    {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e), "errorMessage": "error while generating forecast data" }
                )

    except Exception as e:
        print(e)
        return make_response(
            {"success": False, "message": "Error Occured", "forecast": "a", "error": "${}".format(e) }
        )


if __name__ == "__main__":
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
    app.run(debug=True, host="0.0.0.0", port=5000)
    
