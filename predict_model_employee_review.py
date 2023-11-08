
import pickle
from flask import Flask
from flask import request
from flask import jsonify

# load the model
output_file = 'model.bin'
with open(output_file, "rb") as f_in:
    (dv,model) = pickle.load(f_in)

app = Flask("happiness")
@app.route("/predict", methods = ["POST"])
def predict():
    employee = request.get_json()
    if not employee:
        return jsonify({"error": "No JSON data received"})
    else:
        threshold = 0.5
        X = dv.transform([employee])
        happiness_probability = model.predict_proba(X)[0][1]
        is_happy = happiness_probability>=threshold

        #result = {"message": "Data received", "data": employee}
        result = {
            #"happiness": np.round(happiness_probability*100,2),
            "happiness": happiness_probability,
            "is_happy": bool(is_happy)
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port = 9696)  # not specifying host="an_ip_address" makes it run in localhost (127.0.0.1)
    # the url would be:  http://localhost:9696/predict   
    # we could make requests using the requests module in a different script

