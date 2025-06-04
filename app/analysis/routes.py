import os
import uuid
from flask import (
    render_template, request, redirect, url_for, current_app, jsonify, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from. import analysis_bp
from app.models import db, Analysis
from tasks import process_analysis_step_task # Impor tugas Celery
from app import socketio # Impor socketio dari app package

# Daftar 18 langkah (contoh, perlu disesuaikan)
# Format: {'id': id_langkah, 'name': 'nama_internal_langkah', 'display_name': 'Nama Tampilan Langkah', 'params_needed': ['nama_parameter_opsional']}
# 'params_needed' adalah daftar parameter yang mungkin dibutuhkan dari pengguna untuk langkah ini
ANALYSIS_STEPS =},
    {'id': 3, 'name': 'sift', 'display_name': '3. Analisis Fitur SIFT'},
    {'id': 4, 'name': 'thesis_copy_move', 'display_name': '4. Deteksi Copy-Move (Tesis)'},
    {'id': 5, 'name': 'thesis_splicing', 'display_name': '5. Deteksi Splicing (Tesis)'},
    {'id': 6, 'name': 'thesis_kmeans', 'display_name': '6. Analisis K-Means (Tesis)'},
    {'id': 7, 'name': 'thesis_localization', 'display_name': '7. Lokalisasi Pemalsuan (Tesis)'},
    # Tambahkan 11 langkah lainnya di sini sesuai kebutuhan Anda
    # Pastikan 'name' unik dan digunakan secara konsisten di tasks.py
    {'id': 8, 'name': 'step_8_example', 'display_name': '8. Langkah Contoh 8'},
    {'id': 9, 'name': 'step_9_example', 'display_name': '9. Langkah Contoh 9'},
    {'id': 10, 'name': 'step_10_example', 'display_name': '10. Langkah Contoh 10'},
    {'id': 11, 'name': 'step_11_example', 'display_name': '11. Langkah Contoh 11'},
    {'id': 12, 'name': 'step_12_example', 'display_name': '12. Langkah Contoh 12'},
    {'id': 13, 'name': 'step_13_example', 'display_name': '13. Langkah Contoh 13'},
    {'id': 14, 'name': 'step_14_example', 'display_name': '14. Langkah Contoh 14'},
    {'id': 15, 'name': 'step_15_example', 'display_name': '15. Langkah Contoh 15'},
    {'id': 16, 'name': 'step_16_example', 'display_name': '16. Langkah Contoh 16'},
    {'id': 17, 'name': 'step_17_example', 'display_name': '17. Langkah Contoh 17'},
    {'id': 18, 'name': 'final_review', 'display_name': '18. Tinjauan Akhir & Hasil'},
]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1).[1]lower() in current_app.config # Akses dari config

@analysis_bp.route('/upload', methods=) # Tambahkan GET dan POST
def upload_image_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Tidak ada bagian file', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Tidak ada file yang dipilih', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            original_filename = secure_filename(file.filename)
            file_extension = original_filename.rsplit('.', 1).[1]lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            
            upload_folder = current_app.config # Akses dari config
            stored_image_path = os.path.join(upload_folder, unique_filename)
            file.save(stored_image_path)

            results_base_dir = current_app.config # Akses dari config
            analysis_results_dir_name = str(uuid.uuid4())
            analysis_results_path_abs = os.path.join(results_base_dir, analysis_results_dir_name)
            os.makedirs(analysis_results_path_abs, exist_ok=True)

            new_analysis = Analysis(
                original_image_filename=original_filename,
                stored_image_path=stored_image_path,
                results_directory=analysis_results_dir_name,
                status='uploaded',
                current_step_number=0, # Mulai dari 0, langkah pertama adalah 1
                total_steps=len(ANALYSIS_STEPS)
            )
            db.session.add(new_analysis)
            db.session.commit()
            
            flash('File berhasil diunggah. Memulai analisis...', 'success')
            return redirect(url_for('analysis.process_step_page', analysis_id=new_analysis.id, step_id=1))
        else:
            flash('Jenis file tidak diizinkan', 'error')
            return redirect(request.url)
    return render_template('upload.html', title="Unggah Gambar")

@analysis_bp.route('/process/<int:analysis_id>/step/<int:step_id>', methods=) # Tambahkan GET dan POST
def process_step_page(analysis_id, step_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    if not (1 <= step_id <= len(ANALYSIS_STEPS)):
        flash('ID langkah tidak valid.', 'error')
        return redirect(url_for('main.index')) # Asumsi main.index ada atau ganti dengan halaman yang sesuai

    current_step_info = next((step for step in ANALYSIS_STEPS if step['id'] == step_id), None)
    if not current_step_info:
        flash('Informasi langkah tidak ditemukan.', 'error')
        return redirect(url_for('main.index'))

    task_id_for_template = None # Inisialisasi task_id_for_template

    if request.method == 'POST':
        params_for_task = {}
        if current_step_info.get('params_needed'): # Gunakan.get() untuk keamanan
            for param_name in current_step_info['params_needed']:
                if param_name in request.form:
                    if param_name == 'quality':
                         try:
                            params_for_task[param_name] = int(request.form[param_name])
                         except ValueError:
                            flash(f'Nilai tidak valid untuk {param_name}. Harus angka.', 'error')
                            return render_template('processing_step.html',
                                                   title=f"Analisis Langkah {step_id}",
                                                   analysis=analysis,
                                                   current_step_info=current_step_info,
                                                   total_steps=len(ANALYSIS_STEPS),
                                                   steps_list=ANALYSIS_STEPS,
                                                   task_id=task_id_for_template) # Kirim task_id
                    else:
                        params_for_task[param_name] = request.form[param_name]
        
        task = process_analysis_step_task.delay(
            analysis_id=analysis.id,
            step_name=current_step_info['name'],
            image_path=analysis.stored_image_path,
            params=params_for_task
        )
        task_id_for_template = task.id # Simpan task_id untuk dikirim ke template
        
        analysis.status = f'processing_{current_step_info["name"]}'
        analysis.current_step_number = step_id
        db.session.commit()
        
        flash(f'Memproses {current_step_info["display_name"]}... ID Tugas: {task.id}', 'info')
        # Tetap di halaman yang sama untuk melihat progres
        return render_template('processing_step.html',
                               title=f"Analisis Langkah {step_id}",
                               analysis=analysis,
                               current_step_info=current_step_info,
                               total_steps=len(ANALYSIS_STEPS),
                               task_id=task_id_for_template, # Kirim task_id ke template
                               steps_list=ANALYSIS_STEPS)

    # Untuk metode GET
    # Logika untuk memeriksa apakah langkah sebelumnya sudah selesai bisa disederhanakan
    # atau ditangani lebih baik dengan status yang lebih granular di model Analysis.
    # Untuk saat ini, kita asumsikan pengguna mengikuti alur atau me-refresh.

    template_name = current_step_info.get('template', 'processing_step.html')
    return render_template(template_name,
                           title=f"Analisis Langkah {step_id}",
                           analysis=analysis,
                           current_step_info=current_step_info,
                           total_steps=len(ANALYSIS_STEPS),
                           steps_list=ANALYSIS_STEPS,
                           task_id=task_id_for_template) # Kirim task_id (akan None pada GET awal)


@analysis_bp.route('/results/<int:analysis_id>')
def analysis_results_page(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    results_data = {
        'original_image_url': url_for('analysis.serve_uploaded_image', filename=os.path.basename(analysis.stored_image_path)),
        'metadata': json.loads(analysis.metadata_result) if analysis.metadata_result else None,
        'ela_image_url': url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=analysis.ela_result_path) if analysis.ela_result_path else None,
        'sift_image_url': url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=analysis.sift_result_path) if analysis.sift_result_path else None,
    }
    
    # Memproses hasil tesis yang disimpan sebagai JSON
    if analysis.thesis_copy_move_result:
        try:
            cm_data = json.loads(analysis.thesis_copy_move_result)
            results_data['thesis_copy_move_details'] = cm_data
            if isinstance(cm_data, dict) and cm_data.get('mask_image_path'):
                results_data['thesis_copy_move_image_url'] = url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=cm_data['mask_image_path'])
        except json.JSONDecodeError:
            results_data['thesis_copy_move_details'] = {"error": "Format JSON tidak valid"}

    if analysis.thesis_splicing_result:
        try:
            sp_data = json.loads(analysis.thesis_splicing_result)
            results_data['thesis_splicing_details'] = sp_data
            if isinstance(sp_data, dict) and sp_data.get('heatmap_image_path'):
                results_data['thesis_splicing_image_url'] = url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=sp_data['heatmap_image_path'])
        except json.JSONDecodeError:
            results_data['thesis_splicing_details'] = {"error": "Format JSON tidak valid"}

    if analysis.thesis_kmeans_result:
        try:
            km_data = json.loads(analysis.thesis_kmeans_result)
            results_data['thesis_kmeans_details'] = km_data
            if isinstance(km_data, dict) and km_data.get('segmented_image_path'):
                results_data['thesis_kmeans_image_url'] = url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=km_data['segmented_image_path'])
        except json.JSONDecodeError:
            results_data['thesis_kmeans_details'] = {"error": "Format JSON tidak valid"}

    if analysis.thesis_localization_result:
        try:
            loc_data = json.loads(analysis.thesis_localization_result)
            results_data['thesis_localization_details'] = loc_data
            if isinstance(loc_data, dict) and loc_data.get('localized_image_path'):
                results_data['thesis_localization_image_url'] = url_for('analysis.serve_analysis_result', analysis_folder=analysis.results_directory, filename=loc_data['localized_image_path'])
        except json.JSONDecodeError:
            results_data['thesis_localization_details'] = {"error": "Format JSON tidak valid"}
            
    return render_template('results.html', title="Hasil Analisis", analysis=analysis, results_data=results_data)

@analysis_bp.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(current_app.config, filename)

@analysis_bp.route('/analysis_results/<path:analysis_folder>/<filename>')
def serve_analysis_result(analysis_folder, filename):
    # Pastikan analysis_folder dan filename aman dan tidak mengarah ke luar direktori yang diizinkan
    # Ini adalah contoh sederhana; validasi path yang lebih kuat mungkin diperlukan
    safe_analysis_folder = secure_filename(analysis_folder) # Membersihkan nama folder
    safe_filename = secure_filename(filename) # Membersihkan nama file

    results_dir = os.path.join(current_app.config, safe_analysis_folder)
    return send_from_directory(results_dir, safe_filename)


@analysis_bp.route('/task_status/<task_id>')
def task_status(task_id):
    task = process_analysis_step_task.AsyncResult(task_id)
    response_data = {'state': task.state}
    if task.state == 'PENDING':
        response_data['status'] = 'Menunggu...'
        response_data['progress'] = 0
    elif task.state!= 'FAILURE':
        response_data['status'] = task.info.get('status', '') if isinstance(task.info, dict) else ''
        response_data['progress'] = task.info.get('progress', 0) if isinstance(task.info, dict) else 0
        if task.state == 'SUCCESS':
            response_data['result'] = task.info.get('result', None) if isinstance(task.info, dict) else None
        if isinstance(task.info, dict) and 'step' in task.info:
            response_data['step'] = task.info['step']
    else:
        response_data['status'] = 'Gagal'
        response_data['error'] = str(task.info) # task.info berisi exception
        response_data['progress'] = 0
    return jsonify(response_data)

# Handler SocketIO (tetap sama)
@socketio.on('connect')
def handle_connect():
    print(f'Klien terhubung: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Klien terputus: {request.sid}')

@socketio.on('join_room')
def handle_join_room_event(data):
    analysis_id = data.get('analysis_id')
    if analysis_id:
        room_name = str(analysis_id)
        socketio.join_room(room_name, sid=request.sid) # Tentukan sid untuk join_room
        print(f"Klien {request.sid} bergabung ke room {room_name}")
        analysis = Analysis.query.get(analysis_id)
        if analysis:
             socketio.emit('initial_status', {
                'analysis_id': analysis.id,
                'status': analysis.status,
                'current_step': analysis.current_step_number,
                'message': f"Terhubung ke analisis ID {analysis.id}. Status saat ini: {analysis.status}"
            }, room=room_name, to=request.sid) # Kirim hanya ke klien yang baru bergabung

@analysis_bp.route('/export/<int:analysis_id>/<string:format_type>')
def export_results(analysis_id, format_type):
    analysis = Analysis.query.get_or_404(analysis_id)
    # Implementasi ekspor (placeholder)
    flash(f'Fungsi ekspor {format_type.upper()} belum diimplementasikan.', 'info')
    return redirect(url_for('analysis.analysis_results_page', analysis_id=analysis_id))