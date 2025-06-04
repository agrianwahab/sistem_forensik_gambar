import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
from celery import Task

from tasks import celery_app as an_celery_app
from config import Config

db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO()

def celery_init_app(app: Flask) -> 'Celery':
    an_celery_app.config_from_object(app.config, namespace="CELERY")

    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    an_celery_app.Task = FlaskTask
    app.extensions["celery"] = an_celery_app
    return an_celery_app

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app, async_mode='eventlet')
    app.extensions["socketio"] = socketio

    celery_init_app(app)

    from .main.routes import main_bp
    app.register_blueprint(main_bp)

    from .analysis.routes import analysis_bp
    app.register_blueprint(analysis_bp, url_prefix='/analysis')

    from .history.routes import history_bp
    app.register_blueprint(history_bp, url_prefix='/history')

    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Create and store results directory path
    results_base_dir = os.path.join(app.static_folder, 'analysis_results')
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
    app.config['RESULTS_BASE_DIR'] = results_base_dir

    return app

from . import models