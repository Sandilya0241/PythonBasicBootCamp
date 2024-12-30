from flask import Flask, render_template,request

'''
It creates an instance of the Flask class, which will be our WSGI (Web Server Gateway Interface) application.
'''

## WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return "<html><h1>This is my application!!!</h1></html>"


@app.route("/index",methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/form",methods=["GET","POST"])
def form():
    if request.method == "POST":
        name = request.form["name"]
        return f'Welcome {name}'
    return render_template("form.html")


'''Execution start for any py file'''
if __name__ == "__main__":
    app.run(debug=True)
