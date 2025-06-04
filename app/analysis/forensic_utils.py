import os
import json
from PIL import Image, ImageChops, ImageEnhance
import cv2 # OpenCV
# Hapus impor current_app dari sini jika tidak digunakan di level modul
# from flask import current_app # Akan diimpor di dalam fungsi jika perlu

#... (placeholder untuk fungsi skrip tesis Anda tetap sama)...
# --- Placeholder untuk metadata2.py ---
def extract_image_metadata_from_script(image_path):
    print(f" Memanggil extract_image_metadata_from_script untuk {image_path}")
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        metadata = {}
        if exif_data:
            from PIL.ExifTags import TAGS # Impor di dalam fungsi jika jarang digunakan
            for k, v in exif_data.items():
                if k in TAGS:
                    metadata] = v # Gunakan nama tag yang bisa dibaca
                else:
                    metadata[k] = v
        # Konversi byte menjadi string jika ada untuk serialisasi JSON
        for key, value in metadata.items():
            if isinstance(value, bytes):
                try:
                    metadata[key] = value.decode('utf-8', errors='replace')
                except Exception:
                     metadata[key] = repr(value) # Fallback jika decode gagal
        return metadata if metadata else {"message": "Tidak ada metadata EXIF ditemukan."}
    except Exception as e:
        return {"error": f"Gagal mengekstrak metadata: {str(e)}"}

# --- Placeholder untuk fungsi-fungsi dari baru5.py ---
def detect_copy_move_from_script(image_path, output_dir):
    print(f" Memanggil detect_copy_move_from_script untuk {image_path}")
    dummy_mask_filename = "copy_move_mask_dummy.png"
    dummy_mask_path = os.path.join(output_dir, dummy_mask_filename)
    try:
        img = Image.open(image_path)
        mask = Image.new('RGB', img.size, color = 'red')
        mask.save(dummy_mask_path)
        # Kembalikan path relatif terhadap RESULTS_BASE_DIR atau nama file saja
        return {"mask_image_path": dummy_mask_filename, "confidence": 0.0, "message": "Implementasi CMFD dari baru5.py belum terintegrasi."}
    except Exception as e:
        return {"error": f"Gagal menjalankan deteksi copy-move: {str(e)}"}


def detect_splicing_from_script(image_path, output_dir):
    print(f" Memanggil detect_splicing_from_script untuk {image_path}")
    dummy_heatmap_filename = "splicing_heatmap_dummy.png"
    dummy_heatmap_path = os.path.join(output_dir, dummy_heatmap_filename)
    try:
        img = Image.open(image_path)
        heatmap = Image.new('RGB', img.size, color = 'blue')
        heatmap.save(dummy_heatmap_path)
        return {"heatmap_image_path": dummy_heatmap_filename, "regions_detected": 0, "message": "Implementasi deteksi splicing dari baru5.py belum terintegrasi."}
    except Exception as e:
        return {"error": f"Gagal menjalankan deteksi splicing: {str(e)}"}

def run_kmeans_analysis_from_script(image_path, output_dir):
    print(f" Memanggil run_kmeans_analysis_from_script untuk {image_path}")
    dummy_kmeans_filename = "kmeans_result_dummy.png"
    dummy_kmeans_path = os.path.join(output_dir, dummy_kmeans_filename)
    try:
        img = Image.open(image_path)
        kmeans_img = Image.new('RGB', img.size, color = 'green')
        kmeans_img.save(dummy_kmeans_path)
        return {"segmented_image_path": dummy_kmeans_filename, "clusters_info": {}, "message": "Implementasi K-Means dari baru5.py belum terintegrasi."}
    except Exception as e:
        return {"error": f"Gagal menjalankan analisis K-Means: {str(e)}"}


def localize_tampering_from_script(image_path, output_dir):
    print(f" Memanggil localize_tampering_from_script untuk {image_path}")
    dummy_localization_filename = "localization_result_dummy.png"
    dummy_localization_path = os.path.join(output_dir, dummy_localization_filename)
    try:
        img = Image.open(image_path)
        localized_img = Image.new('RGB', img.size, color = 'yellow')
        localized_img.save(dummy_localization_path)
        return {"localized_image_path": dummy_localization_filename, "tampered_regions":, "message": "Implementasi lokalisasi tampering dari baru5.py belum terintegrasi."}
    except Exception as e:
        return {"error": f"Gagal menjalankan lokalisasi tampering: {str(e)}"}


# --- Fungsi Pembungkus untuk Tugas Celery ---
def get_analysis_output_dir(analysis_id):
    """Mendapatkan path direktori output absolut untuk analisis tertentu."""
    from app.models import Analysis # Impor di dalam fungsi
    from flask import current_app # Impor di dalam fungsi

    analysis_record = Analysis.query.get(analysis_id)
    
    # Akses RESULTS_BASE_DIR dari current_app.config
    # RESULTS_BASE_DIR adalah path absolut ke static/analysis_results/
    results_base_dir = current_app.config

    if not analysis_record or not analysis_record.results_directory:
        # Jika record tidak ada atau direktori hasil belum diset (seharusnya sudah saat unggah)
        # Buat nama folder unik sebagai fallback
        dir_name = str(analysis_id) # Atau gunakan UUID baru jika analysis_id tidak bisa jadi nama folder
        current_app.logger.warning(f"Record analisis {analysis_id} tidak ditemukan atau results_directory kosong. Menggunakan fallback dir_name: {dir_name}")
    else:
        # analysis_record.results_directory adalah nama folder unik (misalnya UUID)
        dir_name = analysis_record.results_directory
    
    output_dir_abs = os.path.join(results_base_dir, dir_name)
    os.makedirs(output_dir_abs, exist_ok=True)
    return output_dir_abs # Kembalikan path absolut


def extract_metadata_analysis(image_path):
    print(f"Memulai ekstraksi metadata untuk: {image_path}")
    metadata = extract_image_metadata_from_script(image_path)
    return metadata

def run_ela_analysis(image_path: str, quality_level: int = 90, analysis_id=None) -> str:
    print(f"Memulai ELA untuk: {image_path} dengan kualitas {quality_level}")
    if analysis_id is None:
        raise ValueError("analysis_id diperlukan untuk menentukan direktori output ELA.")
    try:
        original_image = Image.open(image_path).convert('RGB')
        
        temp_filename = f"temp_ela_q{quality_level}.jpg" # Nama file sementara yang lebih unik
        original_image.save(temp_filename, "JPEG", quality=quality_level)
        resaved_image = Image.open(temp_filename)
        
        ela_image = ImageChops.difference(original_image, resaved_image)
        
        extrema = ela_image.getextrema()
        max_diff = 0
        for i in range(len(extrema)):
             max_diff = max(max_diff, extrema[i][1])

        scale = 50.0 # Default scale factor, bisa disesuaikan
        if max_diff!= 0:
            scale = min(255.0 / max_diff, scale) # Batasi skala

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        os.remove(temp_filename)

        output_dir_abs = get_analysis_output_dir(analysis_id)
        # Buat nama file yang lebih bersih dan unik
        base_img_name = os.path.splitext(os.path.basename(image_path))
        ela_filename = f"ela_q{quality_level}_{base_img_name}.png"
        ela_output_path_abs = os.path.join(output_dir_abs, ela_filename)
        ela_image.save(ela_output_path_abs)
        
        print(f"ELA selesai, hasil disimpan di: {ela_output_path_abs}")
        # Kembalikan hanya nama file, karena path folder sudah ada di analysis_record.results_directory
        return ela_filename 
    except Exception as e:
        # Log error
        from flask import current_app
        current_app.logger.error(f"Error dalam ELA untuk {image_path}: {e}", exc_info=True)
        raise

def run_sift_analysis(image_path: str, analysis_id=None) -> dict:
    print(f"Memulai SIFT untuk: {image_path}")
    if analysis_id is None:
        raise ValueError("analysis_id diperlukan untuk menentukan direktori output SIFT.")
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak dapat dibaca dari path: {image_path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        try:
            sift = cv2.SIFT_create()
        except AttributeError:
            try:
                sift = cv2.xfeatures2d.SIFT_create() # Untuk versi OpenCV yang lebih lama
            except AttributeError:
                 error_msg = "SIFT tidak tersedia di instalasi OpenCV Anda."
                 from flask import current_app
                 current_app.logger.error(error_msg)
                 return {"error": error_msg, "image_path": None, "keypoints_count": 0}

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        img_with_keypoints = cv2.drawKeypoints(gray, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        output_dir_abs = get_analysis_output_dir(analysis_id)
        base_img_name = os.path.splitext(os.path.basename(image_path))
        sift_filename = f"sift_{base_img_name}.png"
        sift_output_path_abs = os.path.join(output_dir_abs, sift_filename)
        cv2.imwrite(sift_output_path_abs, img_with_keypoints)
        
        keypoints_data =
        #... (logika keypoints_data tetap sama)...
        
        print(f"SIFT selesai, hasil disimpan di: {sift_output_path_abs}")
        return {
            "image_path": sift_filename, # Kembalikan nama file
            "keypoints_count": len(keypoints) if keypoints is not None else 0,
            # "keypoints_data": keypoints_data # Pertimbangkan ukuran data ini
        }
    except Exception as e:
        from flask import current_app
        current_app.logger.error(f"Error dalam SIFT untuk {image_path}: {e}", exc_info=True)
        raise

# Fungsi pembungkus lainnya juga harus mengembalikan nama file, bukan path absolut
def detect_copy_move_from_thesis(image_path: str, analysis_id=None):
    output_dir_abs = get_analysis_output_dir(analysis_id)
    # Fungsi skrip tesis harus diubah untuk menerima output_dir_abs dan mengembalikan nama file
    result = detect_copy_move_from_script(image_path, output_dir_abs)
    # Asumsikan result['mask_image_path'] sekarang adalah nama file
    return result

def detect_splicing_from_thesis(image_path: str, analysis_id=None):
    output_dir_abs = get_analysis_output_dir(analysis_id)
    result = detect_splicing_from_script(image_path, output_dir_abs)
    return result

def run_kmeans_analysis_from_thesis(image_path: str, analysis_id=None):
    output_dir_abs = get_analysis_output_dir(analysis_id)
    result = run_kmeans_analysis_from_script(image_path, output_dir_abs)
    return result

def localize_tampering_from_thesis(image_path: str, analysis_id=None):
    output_dir_abs = get_analysis_output_dir(analysis_id)
    result = localize_tampering_from_script(image_path, output_dir_abs)
    return result