from dotenv import load_dotenv
import os 
from flask import Flask, jsonify, make_response, request
from functools import wraps
import jwt

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


@app.route('/get-forecast', methods=["POST"])
@token_required
def makeForecast(data):
    return make_response({"success": True, "message": "Welcome to forecasting route"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)