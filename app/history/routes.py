from flask import render_template, redirect, url_for, flash
from. import history_bp
from app.models import Analysis

@history_bp.route('/')
def list_history():
    page = request.args.get('page', 1, type=int)
    per_page = 10 # Jumlah item per halaman
    all_analysis = Analysis.query.order_by(Analysis.analysis_start_time.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('history.html', title="Riwayat Analisis", analyses=all_analysis)

@history_bp.route('/view/<int:analysis_id>')
def view_analysis_detail(analysis_id):
    # Arahkan ke halaman hasil utama yang sudah ada
    return redirect(url_for('analysis.analysis_results_page', analysis_id=analysis_id))