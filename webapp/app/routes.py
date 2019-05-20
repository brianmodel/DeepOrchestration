from flask import request, render_template, redirect
from app import app
from werkzeug.utils import secure_filename
from app.utils import allowed_file


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            print("No file attached in request")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            print("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(file.read())

    return render_template("index.html")

