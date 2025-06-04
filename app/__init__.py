import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_socketio import SocketIO
from celery import Task # Impor Task langsung

# Impor instance celery_app dari tasks.py
# Kita akan menamainya an_celery_app di sini untuk menghindari potensi konflik nama
from tasks import celery_app as an_celery_app

# Impor Config dari direktori root proyek
from config import Config

db = SQLAlchemy()
migrate = Migrate()
socketio = SocketIO()

def celery_init_app(app: Flask) -> 'Celery': # Tambahkan kutip untuk 'Celery' jika Celery belum diimpor di sini
    # Konfigurasikan instance Celery yang diimpor (an_celery_app)
    # dengan pengaturan dari konfigurasi aplikasi Flask
    an_celery_app.config_from_object(app.config, namespace="CELERY")

    class FlaskTask(Task): # Gunakan celery.Task
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context(): # Ini menyediakan konteks aplikasi untuk tugas
                return self.run(*args, **kwargs)

    an_celery_app.Task = FlaskTask # Atur kelas tugas kustom untuk kesadaran konteks
    app.extensions["celery"] = an_celery_app # Simpan di ekstensi aplikasi Flask
    return an_celery_app

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(app, async_mode='eventlet') # atau 'gevent'
    app.extensions["socketio"] = socketio # Daftarkan socketio agar bisa diakses dari current_app di tasks

    # Inisialisasi/konfigurasi Celery dengan konteks aplikasi
    celery_init_app(app)

    # Registrasi Blueprint
    from.main.routes import main_bp
    app.register_blueprint(main_bp)

    from.analysis.routes import analysis_bp
    app.register_blueprint(analysis_bp, url_prefix='/analysis')

    from.history.routes import history_bp
    app.register_blueprint(history_bp, url_prefix='/history')

    # Pastikan folder upload ada dan siapkan direktori hasil
    if not os.path.exists(app.config):
        os.makedirs(app.config)
    
    results_base_dir = os.path.join(app.static_folder, 'analysis_results')
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)
    # Simpan path ini di konfigurasi aplikasi agar bisa diakses di tempat lain
    app.config = results_base_dir

    return app

# Impor model di sini untuk menghindari circular import dengan migrate
from. import models