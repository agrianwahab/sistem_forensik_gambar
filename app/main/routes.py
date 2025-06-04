from flask import render_template, redirect, url_for
from. import main_bp

@main_bp.route('/')
def index():
    # Arahkan langsung ke halaman unggah atau tampilkan halaman utama
    return redirect(url_for('analysis.upload_image_page'))

@main_bp.route('/about')
def about():
    return render_template('about.html') # Anda perlu membuat about.html jika mau