from flask import request, render_template, redirect, url_for, send_from_directory
from app import app, UPLOAD_FOLDER
from werkzeug.utils import secure_filename
from app.utils import allowed_file, process_file
import os


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            print("No file attached in request")
            return redirect(request.url)
        file = request.files["file"]
        text = request.form["instrument"]
        if file.filename == "":
            print("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            process_file(
                os.path.join(app.config["UPLOAD_FOLDER"], filename), filename, text
            )
            return redirect(url_for("uploaded_file", filename=filename))

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["DOWNLOAD_FOLDER"], filename, as_attachment=True
    )
