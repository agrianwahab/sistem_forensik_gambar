import os
import json
from PIL import Image, ImageChops, ImageEnhance
import cv2
from flask import current_app

def get_analysis_output_dir(analysis_id):
    """Mendapatkan path direktori output absolut untuk analisis tertentu."""
    from app.models import Analysis

    analysis_record = Analysis.query.get(analysis_id)
    
    results_base_dir = current_app.config['RESULTS_BASE_DIR']

    if not analysis_record or not analysis_record.results_directory:
        dir_name = str(analysis_id)
        current_app.logger.warning(f"Record analisis {analysis_id} tidak ditemukan atau results_directory kosong. Menggunakan fallback dir_name: {dir_name}")
    else:
        dir_name = analysis_record.results_directory
    
    output_dir_abs = os.path.join(results_base_dir, dir_name)
    os.makedirs(output_dir_abs, exist_ok=True)
    return output_dir_abs

def extract_image_metadata_from_script(image_path):
    print(f" Memanggil extract_image_metadata_from_script untuk {image_path}")
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        metadata = {}
        if exif_data:
            from PIL.ExifTags import TAGS
            for k, v in exif_data.items():
                if k in TAGS:
                    metadata[TAGS[k]] = v
                else:
                    metadata[k] = v
        for key, value in metadata.items():
            if isinstance(value, bytes):
                try:
                    metadata[key] = value.decode('utf-8', errors='replace')
                except Exception:
                     metadata[key] = repr(value)
        return metadata if metadata else {"message": "Tidak ada metadata EXIF ditemukan."}
    except Exception as e:
        return {"error": f"Gagal mengekstrak metadata: {str(e)}"}

def detect_copy_move_from_script(image_path, output_dir):
    print(f" Memanggil detect_copy_move_from_script untuk {image_path}")
    dummy_mask_filename = "copy_move_mask_dummy.png"
    dummy_mask_path = os.path.join(output_dir, dummy_mask_filename)
    try:
        img = Image.open(image_path)
        mask = Image.new('RGB', img.size, color = 'red')
        mask.save(dummy_mask_path)
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
        return {"localized_image_path": dummy_localization_filename, "tampered_regions": [], "message": "Implementasi lokalisasi tampering dari baru5.py belum terintegrasi."}
    except Exception as e:
        return {"error": f"Gagal menjalankan lokalisasi tampering: {str(e)}"}

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
        
        temp_filename = f"temp_ela_q{quality_level}.jpg"
        original_image.save(temp_filename, "JPEG", quality=quality_level)
        resaved_image = Image.open(temp_filename)
        
        ela_image = ImageChops.difference(original_image, resaved_image)
        
        extrema = ela_image.getextrema()
        max_diff = 0
        for i in range(len(extrema)):
             max_diff = max(max_diff, extrema[i][1])

        scale = 50.0
        if max_diff!= 0:
            scale = min(255.0 / max_diff, scale)

        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        os.remove(temp_filename)

        output_dir_abs = get_analysis_output_dir(analysis_id)
        base_img_name = os.path.splitext(os.path.basename(image_path))[0]
        ela_filename = f"ela_q{quality_level}_{base_img_name}.png"
        ela_output_path_abs = os.path.join(output_dir_abs, ela_filename)
        ela_image.save(ela_output_path_abs)
        
        print(f"ELA selesai, hasil disimpan di: {ela_output_path_abs}")
        return ela_filename
    except Exception as e:
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
                sift = cv2.xfeatures2d.SIFT_create()
            except AttributeError:
                 error_msg = "SIFT tidak tersedia di instalasi OpenCV Anda."
                 current_app.logger.error(error_msg)
                 return {"error": error_msg, "image_path": None, "keypoints_count": 0}

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        img_with_keypoints = cv2.drawKeypoints(gray, keypoints, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        output_dir_abs = get_analysis_output_dir(analysis_id)
        base_img_name = os.path.splitext(os.path.basename(image_path))[0]
        sift_filename = f"sift_{base_img_name}.png"
        sift_output_path_abs = os.path.join(output_dir_abs, sift_filename)
        cv2.imwrite(sift_output_path_abs, img_with_keypoints)
        
        print(f"SIFT selesai, hasil disimpan di: {sift_output_path_abs}")
        return {
            "image_path": sift_filename,
            "keypoints_count": len(keypoints) if keypoints is not None else 0,
        }
    except Exception as e:
        current_app.logger.error(f"Error dalam SIFT untuk {image_path}: {e}", exc_info=True)
        raise

def detect_copy_move_from_thesis(image_path: str, analysis_id=None):
    output_dir_abs = get_analysis_output_dir(analysis_id)
    result = detect_copy_move_from_script(image_path, output_dir_abs)
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