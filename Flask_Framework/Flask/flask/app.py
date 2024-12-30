from flask import Flask

'''
It creates an instance of the Flask class, which will be our WSGI (Web Server Gateway Interface) application.
'''

## WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return "Welcome to My app!!"


@app.route("/index")
def index():
    return "This is index page"


'''Execution start for any py file'''
if __name__ == "__main__":
    app.run()
