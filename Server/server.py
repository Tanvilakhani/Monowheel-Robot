from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def handle_post():
    # Extract JSON data from the POST request
    distance = request.get_json()  # If the data is sent in JSON format
    if distance:
        print(distance)
        return jsonify({"received_data": distance})
    
    # Or extract form data (if it's not JSON)
    form_data = request.form
    print(form_data)
    return jsonify({"received_form_data": form_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
