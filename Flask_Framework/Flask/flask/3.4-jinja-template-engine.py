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

from flask import Flask, render_template,request,redirect,url_for

'''
It creates an instance of the Flask class, which will be our WSGI (Web Server Gateway Interface) application.
'''

## WSGI Application
app = Flask(__name__)

@app.route("/")
def welcome():
    return "<html><h1>This is my application!!!</h1></html>"


@app.route("/successres/<int:avgscore>")
def successres(avgscore):
    return render_template("3.4-jinja-template-engine-result.html",avgscore=avgscore)


@app.route("/getresults",methods=['GET','POST'])
def getresults():
    avgscore = 0
    if request.method == "POST":
        Science = float(request.form["Science"])
        Mathematics = float(request.form["Mathematics"])
        CLanguage = float(request.form["CLanguage"])
        DataScience = float(request.form["DataScience"])
        avgscore = (Science + Mathematics + CLanguage + DataScience) / 4
    else:
        return render_template("3.4-jinja-template-engine.html")    
    return redirect(url_for("successres",avgscore=avgscore))


'''
Execution start for any py file
'''
if __name__ == "__main__":
    app.run(debug=True)
