from flask import Blueprint

analysis_bp = Blueprint('analysis', __name__, template_folder='../templates', static_folder='../static')

from. import routes # Impor rute setelah blueprint dibuat