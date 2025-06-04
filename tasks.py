import time
import os
import json
from celery import Celery
from flask import current_app # Gunakan ini di dalam tugas untuk konteks aplikasi

# Ini adalah instance aplikasi Celery yang akan dikonfigurasi oleh create_app.
# Perintah `celery -A tasks.celery_app worker` akan mencari instance ini.
# Nama 'tasks' bisa berupa nama modul atau nama aplikasi Anda.
celery_app = Celery('tasks')

# Impor model dan utilitas lain di dalam fungsi tugas jika memerlukan konteks aplikasi,
# atau jika ada risiko circular import pada level modul.

@celery_app.task(bind=True)
def process_analysis_step_task(self, analysis_id, step_name, image_path, params=None):
    # current_app adalah konteks aplikasi Flask, disediakan oleh FlaskTask.
    # Tidak perlu memanggil create_app() di sini.
    
    # Impor modul spesifik aplikasi yang memerlukan konteks di sini
    from app.models import db, Analysis
    from app.analysis.forensic_utils import (
        run_ela_analysis,
        run_sift_analysis,
        detect_copy_move_from_thesis,
        detect_splicing_from_thesis,
        extract_metadata_analysis,
        run_kmeans_analysis_from_thesis,
        localize_tampering_from_thesis
    )
    
    # Akses SocketIO melalui ekstensi current_app
    socketio = current_app.extensions.get('socketio')

    # --- Sisa logika tugas Anda ---
    # Contoh:
    analysis_record = Analysis.query.get(analysis_id) # Menggunakan db dari konteks current_app
    if not analysis_record:
        self.update_state(state='FAILURE', meta={'step': step_name, 'error': 'Analysis record not found'})
        # Log atau tangani error dengan benar
        return {'step': step_name, 'status': 'FAILURE', 'error': 'Analysis record not found'}

    try:
        self.update_state(state='PROGRESS', meta={'step': step_name, 'status': 'started', 'progress': 0})
        if socketio:
            socketio.emit('progress_update', {
                'task_id': self.request.id,
                'analysis_id': analysis_id,
                'step': step_name,
                'status': 'dimulai',
                'progress': 0
            }, room=str(analysis_id))

        result_data = None
        # Simulasi pemrosesan dan pembaruan progres
        total_sub_steps = 3 # Kurangi untuk pengujian lebih cepat
        for i in range(total_sub_steps):
            time.sleep(0.5) # Kurangi waktu tidur
            progress = int(((i + 1) / total_sub_steps) * 100)
            self.update_state(state='PROGRESS', meta={'step': step_name, 'status': f'memproses... {progress}%', 'progress': progress})
            if socketio:
                socketio.emit('progress_update', {
                    'task_id': self.request.id,
                    'analysis_id': analysis_id,
                    'step': step_name,
                    'status': f'memproses... {progress}%',
                    'progress': progress
                }, room=str(analysis_id))

        # Logika pemanggilan fungsi forensik berdasarkan step_name
        if step_name == 'metadata':
            result_data = extract_metadata_analysis(image_path)
            analysis_record.metadata_result = json.dumps(result_data) if result_data else None
        elif step_name == 'ela':
            quality = params.get('quality', 90) if params else 90
            # Pastikan analysis_id diteruskan ke fungsi utilitas jika ia membutuhkannya untuk path output
            result_path = run_ela_analysis(image_path, quality_level=quality, analysis_id=analysis_id)
            analysis_record.ela_result_path = result_path
            result_data = {'ela_image_path': result_path} # Kirim path absolut atau relatif yang sesuai
        elif step_name == 'sift':
            sift_output = run_sift_analysis(image_path, analysis_id=analysis_id)
            analysis_record.sift_result_path = sift_output.get('image_path') # Ini adalah path absolut
            result_data = sift_output
        elif step_name == 'thesis_copy_move':
            cm_output = detect_copy_move_from_thesis(image_path, analysis_id=analysis_id)
            analysis_record.thesis_copy_move_result = json.dumps(cm_output) if cm_output else None
            result_data = cm_output
        elif step_name == 'thesis_splicing':
            sp_output = detect_splicing_from_thesis(image_path, analysis_id=analysis_id)
            analysis_record.thesis_splicing_result = json.dumps(sp_output) if sp_output else None
            result_data = sp_output
        elif step_name == 'thesis_kmeans':
            km_output = run_kmeans_analysis_from_thesis(image_path, analysis_id=analysis_id)
            analysis_record.thesis_kmeans_result = json.dumps(km_output) if km_output else None
            result_data = km_output
        elif step_name == 'thesis_localization':
            loc_output = localize_tampering_from_thesis(image_path, analysis_id=analysis_id)
            analysis_record.thesis_localization_result = json.dumps(loc_output) if loc_output else None
            result_data = loc_output
        else:
            time.sleep(1) # Simulasi pekerjaan untuk langkah lain
            result_data = {'message': f'Hasil untuk langkah {step_name}'}
            # Anda mungkin ingin menyimpan hasil generik ini ke model Analysis
            # setattr(analysis_record, f'{step_name}_result_json', json.dumps(result_data))


        analysis_record.status = f'completed_{step_name}'
        # Jika ini adalah langkah terakhir, set status keseluruhan
        if step_name == 'final_review': # Asumsikan 'final_review' adalah nama internal langkah ke-18
             analysis_record.status = 'completed_all'
        analysis_record.analysis_end_time = db.func.now() # Set waktu selesai jika langkah ini adalah akhir
        db.session.commit()

        self.update_state(state='SUCCESS', meta={'step': step_name, 'status': 'selesai', 'result': result_data, 'progress': 100})
        if socketio:
            socketio.emit('progress_update', {
                'task_id': self.request.id,
                'analysis_id': analysis_id,
                'step': step_name,
                'status': 'selesai',
                'result': result_data,
                'progress': 100
            }, room=str(analysis_id))
        return {'step': step_name, 'status': 'SUCCESS', 'result': result_data}

    except Exception as e:
        db.session.rollback() # Rollback jika terjadi error sebelum commit
        if analysis_record: # Pastikan analysis_record ada
            analysis_record.status = f'failed_{step_name}'
            analysis_record.error_message = str(e)
            db.session.commit()
        
        # Log error ke konsol Celery atau logger yang lebih baik
        current_app.logger.error(f"Error dalam tugas Celery {step_name} untuk analysis_id {analysis_id}: {e}", exc_info=True)

        self.update_state(state='FAILURE', meta={'step': step_name, 'error': str(e), 'status': 'gagal'})
        if socketio:
             socketio.emit('progress_update', {
                'task_id': self.request.id,
                'analysis_id': analysis_id,
                'step': step_name,
                'status': 'gagal',
                'error': str(e)
            }, room=str(analysis_id))
        raise # Re-raise exception agar Celery menandainya sebagai FAILED