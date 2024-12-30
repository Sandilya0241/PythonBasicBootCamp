###
# 
# HTTP Verbs - PUT & DELETE
# WORKING WITH APIS - JSON
# 
# ###

from flask import Flask, jsonify, request


app = Flask(__name__)


items_data = [
    {
        "id":1,
        "name":"Item 1",
        "description":"This is Item1 description"
    },{
        "id":2,
        "name":"Item 2",
        "description":"This is Item2 description"
    }
]


@app.route("/")
def home():
    return "Welcome to the sample to-do list app!!!"


@app.route("/items")
def get_items():
    return jsonify(items_data)


@app.route("/items/<int:item_id>")
def get_item(item_id):
    item = next((item for item in items_data if items_data["id"] == item_id), None)
    if item is None:
        return jsonify({"error":"Item Not Found"})
    else:
        return jsonify(item)

@app.route("/items",methods=["POST"])
def create_item():
    if not request.json or not "name" in request.json:
        return jsonify({"error":"Invalid Input"})
    new_item = {
        "id":items_data[-1]["id"] + 1 if items_data else 1,
        "name":request.json["name"],
        "description":request.json["description"]
    }
    items_data.append(new_item)
    return jsonify(new_item)


##PUT: Update record
@app.route("/items",methods=["PUT"])
def update_item():
    if not request.json:
        return jsonify({"error":"Invalid Input"})
    else:
        item = next((item for item in items_data if item["id"] == request.json["id"]), None)
        if item is None:
            return jsonify({"error":"Invalid Input"})
        else:
            item["name"] = request.json.get("name",item["name"])
            item["description"] = request.json.get("description",item["description"])
    return jsonify(item)


##PUT: Update record
@app.route("/items/<item_id>",methods=["DELETE"])
def delete_item(item_id):
    global items_data
    items = (item for item in items_data if item["id"] != item_id)
    return f"{item_id} deleted sucessfully"
    

if __name__ == "__main__":
    app.run(debug=True)