##
# 
# 1. Building URL dynamically
# 2. Variable rule
# 3. Jinja2 Template Engine
# 
# #


###
# 
# Jinja 2 Template Engine
# {{  }} = Expressions to print output in HTML
# {%  %} = Conditions, For Loops
# {#  #} = Single line comments
# 
# ###

from flask import Flask, render_template,request

'''
It creates an instance of the Flask class, which will be our WSGI (Web Server Gateway Interface) application.
'''

## WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return "<html><h1>This is my application!!!</h1></html>"

#Variable Rule - Restricting parameter with a datatype
@app.route("/success/<int:score>")
def success(score):
    return "Your score is " + str(score)

'''
Execution start for any py file
'''
if __name__ == "__main__":
    app.run(debug=True)