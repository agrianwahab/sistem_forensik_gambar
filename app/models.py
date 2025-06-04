from. import db # Mengimpor objek db dari app/__init__.py
from sqlalchemy.sql import func
import json

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_image_filename = db.Column(db.String(255), nullable=False)
    # Path ke file asli yang disimpan di server (misalnya, di UPLOAD_FOLDER)
    stored_image_path = db.Column(db.String(255), nullable=False, unique=True)
    # Path ke folder hasil spesifik untuk analisis ini (misalnya, static/analysis_results/<analysis_id>/)
    results_directory = db.Column(db.String(255), nullable=True)

    analysis_start_time = db.Column(db.DateTime(timezone=True), server_default=func.now())
    analysis_end_time = db.Column(db.DateTime(timezone=True), nullable=True)
    
    # Status keseluruhan analisis, atau status langkah terakhir yang berhasil/gagal
    # Contoh: 'uploaded', 'processing_metadata', 'completed_metadata', 'failed_ela', 'completed_all'
    status = db.Column(db.String(100), nullable=False, default='uploaded')
    current_step_number = db.Column(db.Integer, default=0) # Untuk melacak langkah ke-X dari 18
    total_steps = db.Column(db.Integer, default=18) # Total langkah dalam alur kerja

    # Path ke hasil analisis spesifik (bisa berupa path file relatif terhadap results_directory atau JSON string)
    metadata_result = db.Column(db.Text, nullable=True) # JSON string
    ela_result_path = db.Column(db.String(255), nullable=True) # Path ke gambar ELA
    sift_result_path = db.Column(db.String(255), nullable=True) # Path ke gambar SIFT
    
    # Hasil dari skrip tesis (bisa JSON dengan path ke masker & data, atau data langsung)
    thesis_copy_move_result = db.Column(db.Text, nullable=True)
    thesis_splicing_result = db.Column(db.Text, nullable=True)
    thesis_kmeans_result = db.Column(db.Text, nullable=True)
    thesis_localization_result = db.Column(db.Text, nullable=True)
    
    # Kolom untuk menyimpan hasil dari 18 langkah secara lebih dinamis jika diperlukan
    # Misalnya, sebuah kolom JSON yang menyimpan path atau data untuk setiap langkah
    # step_results = db.Column(db.Text, nullable=True) # JSON: {'step_1_name': 'path/to/result',...}

    overall_conclusion = db.Column(db.Text, nullable=True) # Ringkasan kesimpulan analisis
    error_message = db.Column(db.Text, nullable=True) # Pesan jika terjadi kesalahan

    def __repr__(self):
        return f'<Analysis {self.id} - {self.original_image_filename}>'

    def get_step_results(self):
        try:
            return json.loads(self.step_results) if self.step_results else {}
        except json.JSONDecodeError:
            return {}

    def set_step_result(self, step_name, result_data):
        results = self.get_step_results()
        results[step_name] = result_data
        self.step_results = json.dumps(results)