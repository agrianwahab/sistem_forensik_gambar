from flask import Blueprint

history_bp = Blueprint('history', __name__, template_folder='../templates')

from. import routes