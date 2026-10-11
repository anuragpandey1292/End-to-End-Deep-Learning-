from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = FastAPI(title="Store Sales Deep Learning Predictor")

# Optional: If you want to build a simple HTML frontend later, 
# templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_home():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Store Sales Deep Learning MLOps</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 50px; background-color: #f4f4f9; }
                .container { max-width: 500px; padding: 30px; background: white; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
                h2 { color: #333; }
                label { font-weight: bold; display: block; margin-top: 15px; }
                input, select { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
                button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; margin-top: 20px; width: 100%; }
                button:hover { background-color: #45a049; }
                .result { margin-top: 20px; font-size: 18px; font-weight: bold; color: #2c3e50; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Store Sales Forecasting (Deep Learning)</h2>
                <form action="/predict" method="post">
                    <label for="store_nbr">Store Number:</label>
                    <input type="number" id="store_nbr" name="store_nbr" required value="1">

                    <label for="family">Product Family:</label>
                    <input type="text" id="family" name="family" required value="AUTOMOTIVE">

                    <label for="onpromotion">On Promotion Count:</label>
                    <input type="number" id="onpromotion" name="onpromotion" required value="0">

                    <button type="submit">Predict Sales</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict_datapoint(
    store_nbr: int = Form(...),
    family: str = Form(...),
    onpromotion: int = Form(...)
):
    try:
        # 1. Collect form data into CustomData class
        data = CustomData(
            store_nbr=store_nbr,
            family=family,
            onpromotion=onpromotion
        )
        final_new_data = data.get_data_as_data_frame()
        
        # 2. Run prediction pipeline
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(final_new_data)
        
        predicted_value = round(float(prediction[0]), 2)

        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Prediction Result</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 50px; background-color: #f4f4f9; }}
                    .container {{ max-width: 500px; padding: 30px; background: white; border-radius: 8px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }}
                    h2 {{ color: #27ae60; }}
                    a {{ display: inline-block; margin-top: 20px; text-decoration: none; color: #3498db; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Prediction Successful!</h2>
                    <p>Estimated Sales: <strong>{predicted_value}</strong></p>
                    <a href="/">← Make Another Prediction</a>
                </div>
            </body>
        </html>
        """
    except Exception as e:
        return f"<h3>Error occurred:</h3><p>{str(e)}</p><a href='/'>Back</a>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)