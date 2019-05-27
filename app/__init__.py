from flask import Flask
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/uploads/"
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + "/downloads/"
ALLOWED_EXTENSIONS = {"mid", "midi"}

app.config["DOWNLOAD_FOLDER"] = DOWNLOAD_FOLDER
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

from app import routes
