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
from prophet import Prophet
from PIL import Image
import pytesseract


app = Flask(__name__)

# Load Environment Variables
load_dotenv()
JWT_SECRET = os.environ.get("JWT_SECRET")

# Functions


def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        if not token:
            return make_response({"success": False, "message": "Token not found"}, 201)
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except:
            return make_response({"success": False, "message": "Error while decoding JWT"}, 201)

        return f(data, *args, **kwargs)
    return decorator

# Main Routes


@app.route('/')
def index():
    return make_response({"success": True, "message": "Welcome to Invoice Flask Server"})


@app.route('/validate-upload', methods=["POST"])
@token_required
def validateUpload(_):
    requestBody = request.json
    base64Image = requestBody["image"]
    companyName = str(requestBody["companyName"])

    imgstring = base64Image.split('base64,')[-1].strip()
    
    image_string = io.BytesIO(base64.b64decode(imgstring))
    image = Image.open(image_string)

    # pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract\tesseract.exe'
    
    extractedInformation = str(pytesseract.image_to_string(image))

    checkString = extractedInformation.lower()
    companyNameString = companyName.lower()

    deedIndex = checkString.find("deed")
    partnershipIndex = checkString.find("partnership")
    companyNameIndex = checkString.find(companyNameString)

    if (deedIndex != -1 & partnershipIndex != -1 & companyNameIndex != -1):
        return make_response({"success": True, "message": "Image File Validated", "information": extractedInformation})
    else: 
        return make_response({"success": False, "message": "Image File Invalid", "information": extractedInformation})
   
   
@app.route('/get-forecast', methods=["POST"])
@token_required
def makeForecast(data):
    data = request.json
    monthData = None

    if ("monthData" in data == False):
        return make_response({"success": True, "message": "Welcome to forecasting route"})
    else:
        monthData = data["monthData"]

    class ProphetPos(Prophet):

        @staticmethod
        def piecewise_linear(t, deltas, k, m, changepoint_ts):
            """Evaluate the piecewise linear function, keeping the trend
            positive.

            Parameters
            ----------
            t: np.array of times on which the function is evaluated.
            deltas: np.array of rate changes at each changepoint.
            k: Float initial rate.
            m: Float initial offset.
            changepoint_ts: np.array of changepoint times.

            Returns
            -------
            Vector trend(t).
            """
            # Intercept changes
            gammas = -changepoint_ts * deltas
            # Get cumulative slope and intercept at each t
            k_t = k * np.ones_like(t)
            m_t = m * np.ones_like(t)
            for s, t_s in enumerate(changepoint_ts):
                indx = t >= t_s
                k_t[indx] += deltas[s]
                m_t[indx] += gammas[s]
            trend = k_t * t + m_t
            if max(t) <= 1:
                return trend
            # Add additional deltas to force future trend to be positive
            indx_future = np.argmax(t >= 1)
            while min(trend[indx_future:]) < 0:
                indx_neg = indx_future + np.argmax(trend[indx_future:] < 0)
                k_t[indx_neg:] -= k_t[indx_neg]
                m_t[indx_neg:] -= m_t[indx_neg]
                trend = k_t * t + m_t
            return trend

        def predict(self, df=None):
            fcst = super().predict(df=df)
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                fcst[col] = fcst[col].clip(lower=0.0)
            return fcst

    df = json_normalize(monthData)
    df = df.drop(["index"], axis=1)
    df.columns = ["ds", "y"]
    df['ds'] = pd.to_datetime(df['ds'])
    maxValue = df[['y']].max()
    minValue = df[['y']].min()
    df['y'] = np.log(1 + df['y'])
    print(maxValue)
    df["cap"] = maxValue['y'] * 1.3

    model = ProphetPos(seasonality_mode='multiplicative').fit(df)
    future = model.make_future_dataframe(periods=57, freq = 'ms')
    future["cap"] = np.log(maxValue['y'] * 1.3) 
    forecast = model.predict(future)
    model.history['y'] = np.exp(model.history['y']) - 1

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = np.exp(forecast[col]) - 1

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = forecast[col].clip(lower=minValue['y']*0.8, upper=maxValue['y']*1.4)

    foreCastedValues = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    foreCastedValues.tail()

    returnVal = foreCastedValues.to_json(orient='records')

    
    print(foreCastedValues)


    return make_response({"success": True, "message": "Welcome to forecasting route", "data": returnVal})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
