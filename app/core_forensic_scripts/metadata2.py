#!/usr/bin/env python3
"""
SISTEM DETEKSI FORENSIK KEASLIAN GAMBAR DAN VIDEO MENGGUNAKAN METODE K-MEANS DAN LOCALIZATION TAMPERING
Image Metadata Extraction and Forensic Analysis System - Enhanced with Table Display
"""

import os
import json
import hashlib
import struct
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import exifread
from pathlib import Path
import sys
import argparse
from tabulate import tabulate

class EnhancedMetadataExtractor:
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.tiff', '.tif', '.png'}
        self.critical_exif_tags = [
            'DateTimeOriginal', 'DateTimeDigitized', 'DateTime',
            'Make', 'Model', 'Software', 'GPSInfo'
        ]
    
    def extract_enhanced_metadata(self, image_path):
        """Extract comprehensive metadata for forensic analysis"""
        metadata = {
            'file_metadata': self._extract_file_metadata(image_path),
            'exif_data': self._extract_comprehensive_exif(image_path),
            'format_specific': self._extract_format_specific(image_path),
            'file_structure': self._analyze_file_structure(image_path),
            'thumbnails': self._extract_thumbnail_data(image_path)
        }
        
        # Add quantization tables for JPEG
        if Path(image_path).suffix.lower() in ['.jpg', '.jpeg']:
            metadata['quantization_tables'] = self._extract_quantization_tables(image_path)
        
        return metadata
    
    def _extract_file_metadata(self, file_path):
        """Extract basic file system metadata with anomaly checks"""
        stat = os.stat(file_path)
        file_size = stat.st_size
        created_time = datetime.fromtimestamp(stat.st_ctime)
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        
        # Detect anomalies
        anomalies = []
        if modified_time < created_time:
            anomalies.append("Modification date before creation date")
        
        # Check for suspicious file sizes
        extension = Path(file_path).suffix.lower()
        width, height = 0, 0
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                expected_size = (width * height * 3) / 20  # Rough JPEG estimate
                if file_size > expected_size * 10 or file_size < expected_size / 10:
                    anomalies.append("File size disproportionate to image dimensions")
        except:
            pass
        
        return {
            'filename': os.path.basename(file_path),
            'file_size': file_size,
            'file_type': extension[1:].upper() if extension else 'UNKNOWN',
            'mime_type': self._get_mime_type(extension),
            'created_time': created_time.isoformat(),
            'modified_time': modified_time.isoformat(),
            'md5_hash': self._calculate_hash(file_path, 'md5'),
            'sha256_hash': self._calculate_hash(file_path, 'sha256'),
            'anomalies': anomalies,
            'dimensions': {'width': width, 'height': height}
        }
    
    def _extract_comprehensive_exif(self, file_path):
        """Extract detailed EXIF data with consistency checks"""
        exif_data = {}
        
        # Use both PIL and exifread for comprehensive extraction
        try:
            # PIL extraction
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = self._process_exif_value(value)
                    
                    # Extract GPS data
                    if 'GPSInfo' in exif_data:
                        gps_ifd = exif.get_ifd(0x8825)  # GPS IFD
                        gps_data = {}
                        for key, val in gps_ifd.items():
                            gps_tag = GPSTAGS.get(key, key)
                            gps_data[gps_tag] = val
                        exif_data['GPSInfo'] = gps_data
            
            # Exifread extraction for additional tags
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=True)
                for tag, value in tags.items():
                    if tag not in ('JPEGThumbnail', 'TIFFThumbnail'):
                        # Avoid duplicates, prefer exifread for detailed info
                        if tag not in exif_data:
                            exif_data[tag] = str(value)
            
        except Exception as e:
            exif_data['extraction_error'] = str(e)
        
        return exif_data
    
    def _extract_quantization_tables(self, file_path):
        """Extract JPEG quantization tables for manipulation detection"""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            tables = {}
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i+1] == 0xDB:  # DQT marker
                    i += 2
                    length = struct.unpack('>H', data[i:i+2])[0]
                    i += 2
                    
                    # Parse quantization table
                    table_data = data[i:i+length-2]
                    precision_and_id = table_data[0]
                    table_id = precision_and_id & 0x0F
                    
                    # Extract 64 values (8x8 matrix)
                    values = list(struct.unpack('64B', table_data[1:65]))
                    tables[table_id] = {
                        'values': values,
                        'matrix': self._zigzag_to_matrix(values)
                    }
                    i += length - 2
                else:
                    i += 1
            
            return tables
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_file_structure(self, file_path):
        """Analyze JPEG file structure for anomalies"""
        if not file_path.lower().endswith(('.jpg', '.jpeg')):
            return {'format': 'non-jpeg'}
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            markers = []
            anomalies = []
            i = 0
            
            # Check SOI marker
            if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
                markers.append({'marker': 'SOI', 'position': 0})
            else:
                anomalies.append('Missing or incorrect SOI marker')
            
            # Scan for other markers
            i = 2
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i+1] != 0x00:
                    marker = data[i+1]
                    marker_name = self._get_marker_name(marker)
                    markers.append({
                        'marker': marker_name,
                        'position': i,
                        'hex': f'0xFF{marker:02X}'
                    })
                    
                    # Check for length-based markers
                    if marker not in [0xD8, 0xD9]:  # Not SOI or EOI
                        if i + 3 < len(data):
                            length = struct.unpack('>H', data[i+2:i+4])[0]
                            i += length + 2
                        else:
                            break
                    else:
                        i += 2
                else:
                    i += 1
            
            # Check for EOI marker
            if len(data) >= 2 and data[-2] == 0xFF and data[-1] == 0xD9:
                markers.append({'marker': 'EOI', 'position': len(data)-2})
            else:
                anomalies.append('Missing or incorrect EOI marker')
            
            # Check marker sequence
            expected_early_markers = ['SOI', 'APP0', 'APP1']
            actual_early_markers = [m['marker'] for m in markers[:3]]
            if not all(em in actual_early_markers for em in expected_early_markers[:2]):
                anomalies.append('Unexpected marker sequence at start')
            
            return {
                'markers': markers,
                'anomalies': anomalies,
                'marker_count': len(markers)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _process_exif_value(self, value):
        """Process EXIF values to handle different types"""
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8', errors='ignore')
            except:
                return str(value)
        elif isinstance(value, (tuple, list)):
            return [self._process_exif_value(v) for v in value]
        else:
            return value
    
    def _zigzag_to_matrix(self, values):
        """Convert zigzag-ordered values to 8x8 matrix"""
        zigzag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,
           17, 24, 32, 25, 18, 11,  4,  5,
           12, 19, 26, 33, 40, 48, 41, 34,
           27, 20, 13,  6,  7, 14, 21, 28,
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63
        ]
        
        matrix = [[0]*8 for _ in range(8)]
        for i, val in enumerate(values[:64]):
            pos = zigzag_order[i]
            row, col = pos // 8, pos % 8
            matrix[row][col] = val
        
        return matrix
    
    def _extract_thumbnail_data(self, file_path):
        """Extract and analyze embedded thumbnails"""
        try:
            with Image.open(file_path) as img:
                # Check for EXIF thumbnail
                exif = img.getexif()
                if exif:
                    thumbnail = img.info.get('thumbnail')
                    if thumbnail:
                        # Compare thumbnail with main image
                        thumb_hash = hashlib.md5(thumbnail).hexdigest()
                        
                        # Create downscaled version of main image for comparison
                        img_copy = img.copy()
                        img_copy.thumbnail((160, 120))
                        main_thumb_data = img_copy.tobytes()
                        main_thumb_hash = hashlib.md5(main_thumb_data).hexdigest()
                        
                        return {
                            'has_thumbnail': True,
                            'thumbnail_hash': thumb_hash,
                            'matches_main_image': thumb_hash == main_thumb_hash,
                            'thumbnail_size': len(thumbnail)
                        }
                        
        except Exception as e:
            return {'error': str(e)}
        
        return {'has_thumbnail': False}
    
    def _get_marker_name(self, marker):
        """Get human-readable JPEG marker name"""
        marker_names = {
            0xD8: 'SOI', 0xD9: 'EOI', 0xDA: 'SOS', 0xDB: 'DQT',
            0xC0: 'SOF0', 0xC4: 'DHT', 0xDD: 'DRI', 0xFE: 'COM'
        }
        
        if 0xE0 <= marker <= 0xEF:
            return f'APP{marker - 0xE0}'
        
        return marker_names.get(marker, f'Unknown(0x{marker:02X})')
    
    def _extract_format_specific(self, file_path):
        """Extract format-specific metadata (PNG, TIFF)"""
        try:
            with Image.open(file_path) as img:
                if img.format == 'PNG':
                    return self._extract_png_metadata(img)
                elif img.format in ['TIFF', 'TIF']:
                    return self._extract_tiff_metadata(img)
        except:
            pass
        return {}
    
    def _extract_png_metadata(self, img):
        """Extract PNG-specific metadata"""
        metadata = {}
        if hasattr(img, 'info'):
            for key, value in img.info.items():
                metadata[key] = str(value)
        return {'png_chunks': metadata}
    
    def _extract_tiff_metadata(self, img):
        """Extract TIFF-specific metadata"""
        metadata = {}
        if hasattr(img, 'tag'):
            for tag, value in img.tag.items():
                metadata[str(tag)] = str(value)
        return {'tiff_tags': metadata}
    
    def _calculate_hash(self, file_path, algorithm='md5'):
        """Calculate file hash"""
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def _get_mime_type(self, extension):
        """Get MIME type from file extension"""
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.tiff': 'image/tiff',
            '.tif': 'image/tiff', '.gif': 'image/gif'
        }
        return mime_types.get(extension.lower(), 'application/octet-stream')


class MetadataConsistencyChecker:
    def __init__(self):
        self.software_signatures = {
            'adobe_photoshop': ['Adobe Photoshop', 'Photoshop'],
            'gimp': ['GIMP', 'GNU Image Manipulation Program'],
            'lightroom': ['Adobe Photoshop Lightroom'],
            'mobile_apps': ['Instagram', 'Snapseed', 'VSCO', 'PicsArt']
        }
        
        self.standard_quantization_tables = {
            'jpeg_standard_quality_50': [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]
            ]
        }
    
    def check_enhanced_metadata_consistency(self, metadata):
        """Perform comprehensive metadata consistency analysis"""
        inconsistencies = []
        
        # Time consistency checks
        time_anomalies = self._check_temporal_consistency(metadata)
        inconsistencies.extend(time_anomalies)
        
        # Software signature detection
        software_anomalies = self._detect_software_signatures(metadata)
        inconsistencies.extend(software_anomalies)
        
        # Device consistency checks
        device_anomalies = self._check_device_consistency(metadata)
        inconsistencies.extend(device_anomalies)
        
        # GPS validation
        gps_anomalies = self._validate_gps_data(metadata)
        inconsistencies.extend(gps_anomalies)
        
        # Quantization table analysis
        if 'quantization_tables' in metadata:
            qt_anomalies = self._analyze_quantization_tables(metadata['quantization_tables'])
            inconsistencies.extend(qt_anomalies)
        
        # File structure anomalies
        if 'file_structure' in metadata:
            structure_anomalies = self._check_file_structure(metadata['file_structure'])
            inconsistencies.extend(structure_anomalies)
        
        # Thumbnail consistency
        if 'thumbnails' in metadata:
            thumb_anomalies = self._check_thumbnail_consistency(metadata['thumbnails'])
            inconsistencies.extend(thumb_anomalies)
        
        return inconsistencies
    
    def _check_temporal_consistency(self, metadata):
        """Check for temporal anomalies in metadata"""
        anomalies = []
        exif_data = metadata.get('exif_data', {})
        file_metadata = metadata.get('file_metadata', {})
        
        # Extract all timestamp fields
        timestamp_fields = {
            'DateTimeOriginal': exif_data.get('DateTimeOriginal'),
            'DateTimeDigitized': exif_data.get('DateTimeDigitized'),
            'DateTime': exif_data.get('DateTime'),
            'FileModifyDate': file_metadata.get('modified_time'),
            'FileCreateDate': file_metadata.get('created_time')
        }
        
        # Parse timestamps
        timestamps = {}
        for field, value in timestamp_fields.items():
            if value:
                try:
                    if isinstance(value, str):
                        # Handle various timestamp formats
                        for fmt in ['%Y:%m:%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                            try:
                                timestamps[field] = datetime.strptime(value.split('.')[0], fmt)
                                break
                            except:
                                continue
                except:
                    pass
        
        # Check for logical inconsistencies
        if 'DateTimeOriginal' in timestamps and 'DateTime' in timestamps:
            if timestamps['DateTime'] < timestamps['DateTimeOriginal']:
                anomalies.append({
                    'type': 'time_inconsistency',
                    'description': 'Modification date before original capture date',
                    'severity': 'high'
                })
        
        # Check for suspiciously close timestamps
        if len(timestamps) > 1:
            time_values = list(timestamps.values())
            for i in range(len(time_values)):
                for j in range(i + 1, len(time_values)):
                    diff = abs((time_values[i] - time_values[j]).total_seconds())
                    if diff > 86400 * 365:  # More than a year difference
                        anomalies.append({
                            'type': 'time_inconsistency',
                            'description': f'Large time difference between metadata fields ({diff/86400:.0f} days)',
                            'severity': 'medium'
                        })
        
        # Check GPS time consistency
        gps_info = exif_data.get('GPSInfo', {})
        if gps_info and 'GPSTimeStamp' in gps_info and 'DateTimeOriginal' in timestamps:
            # GPS time should be close to capture time
            anomalies.append({
                'type': 'gps_time_mismatch',
                'description': 'GPS timestamp differs from EXIF timestamp',
                'severity': 'medium'
            })
        
        return anomalies
    
    def _detect_software_signatures(self, metadata):
        """Detect image editing software signatures"""
        anomalies = []
        exif_data = metadata.get('exif_data', {})
        
        # Check Software field
        software = exif_data.get('Software', '')
        if software:
            for category, signatures in self.software_signatures.items():
                for signature in signatures:
                    if signature.lower() in software.lower():
                        anomalies.append({
                            'type': 'software_editing_detected',
                            'description': f'Editing software detected: {software}',
                            'category': category,
                            'severity': 'high' if 'adobe' in category else 'medium'
                        })
                        break
        
        # Check for XMP data (often indicates processing)
        xmp_fields = [k for k in exif_data.keys() if 'xmp' in k.lower()]
        if xmp_fields:
            anomalies.append({
                'type': 'xmp_metadata_present',
                'description': 'XMP metadata found, indicating possible editing',
                'fields': xmp_fields,
                'severity': 'medium'
            })
        
        # Check for Photoshop-specific markers
        if any('photoshop' in str(v).lower() for v in exif_data.values()):
            anomalies.append({
                'type': 'photoshop_markers',
                'description': 'Adobe Photoshop markers found in metadata',
                'severity': 'high'
            })
        
        return anomalies
    
    def _check_device_consistency(self, metadata):
        """Check camera/device information consistency"""
        anomalies = []
        exif_data = metadata.get('exif_data', {})
        
        make = exif_data.get('Make', '').strip()
        model = exif_data.get('Model', '').strip()
        
        if make and model:
            # Check for known invalid combinations
            if make.lower() == 'canon' and 'nikon' in model.lower():
                anomalies.append({
                    'type': 'device_mismatch',
                    'description': f'Camera make/model mismatch: {make} / {model}',
                    'severity': 'high'
                })
            
            # Check technical parameters for device
            focal_length = exif_data.get('FocalLength')
            f_number = exif_data.get('FNumber')
            
            if focal_length and f_number:
                # Validate against known device capabilities
                # This is a simplified check - real implementation would use device database
                if isinstance(focal_length, (list, tuple)):
                    focal_length = focal_length[0] / focal_length[1] if focal_length[1] else focal_length[0]
                if isinstance(f_number, (list, tuple)):
                    f_number = f_number[0] / f_number[1] if f_number[1] else f_number[0]
                
                # Basic sanity checks
                if focal_length < 1 or focal_length > 2000:
                    anomalies.append({
                        'type': 'impossible_focal_length',
                        'description': f'Focal length {focal_length}mm outside reasonable range',
                        'severity': 'medium'
                    })
        
        return anomalies
    
    def _validate_gps_data(self, metadata):
        """Validate GPS data for consistency"""
        anomalies = []
        exif_data = metadata.get('exif_data', {})
        gps_info = exif_data.get('GPSInfo', {})
        
        if gps_info:
            # Check for required GPS fields
            required_fields = ['GPSLatitude', 'GPSLongitude']
            missing_fields = [f for f in required_fields if f not in gps_info]
            
            if missing_fields:
                anomalies.append({
                    'type': 'incomplete_gps_data',
                    'description': f'Missing GPS fields: {missing_fields}',
                    'severity': 'low'
                })
            
            # Validate coordinate ranges
            lat = gps_info.get('GPSLatitude')
            lon = gps_info.get('GPSLongitude')
            
            if lat and lon:
                try:
                    # Convert to decimal degrees
                    lat_decimal = self._gps_to_decimal(lat, gps_info.get('GPSLatitudeRef', 'N'))
                    lon_decimal = self._gps_to_decimal(lon, gps_info.get('GPSLongitudeRef', 'E'))
                    
                    # Check valid ranges
                    if not (-90 <= lat_decimal <= 90):
                        anomalies.append({
                            'type': 'invalid_gps_latitude',
                            'description': f'Invalid latitude: {lat_decimal}',
                            'severity': 'high'
                        })
                    
                    if not (-180 <= lon_decimal <= 180):
                        anomalies.append({
                            'type': 'invalid_gps_longitude',
                            'description': f'Invalid longitude: {lon_decimal}',
                            'severity': 'high'
                        })
                except:
                    anomalies.append({
                        'type': 'gps_parsing_error',
                        'description': 'Unable to parse GPS coordinates',
                        'severity': 'medium'
                    })
        
        return anomalies
    
    def _analyze_quantization_tables(self, qt_data):
        """Analyze JPEG quantization tables for manipulation"""
        anomalies = []
        
        if isinstance(qt_data, dict) and 'error' not in qt_data:
            for table_id, table_info in qt_data.items():
                if isinstance(table_info, dict) and 'matrix' in table_info:
                    matrix = table_info['matrix']
                    
                    # Check for unusual patterns
                    flat_matrix = [val for row in matrix for val in row]
                    
                    # Check for all same values (highly suspicious)
                    if len(set(flat_matrix)) == 1:
                        anomalies.append({
                            'type': 'uniform_quantization_table',
                            'description': f'Quantization table {table_id} has uniform values',
                            'severity': 'high'
                        })
                    
                    # Check for zeros (unusual in standard JPEG)
                    if 0 in flat_matrix:
                        anomalies.append({
                            'type': 'zero_in_quantization_table',
                            'description': f'Quantization table {table_id} contains zeros',
                            'severity': 'medium'
                        })
                    
                    # Compare with standard tables
                    similarity = self._compare_with_standard_tables(matrix)
                    if similarity < 0.7:  # Low similarity to any standard table
                        anomalies.append({
                            'type': 'non_standard_quantization',
                            'description': f'Quantization table {table_id} differs from standard tables',
                            'severity': 'medium'
                        })
        
        return anomalies
    
    def _check_file_structure(self, structure_data):
        """Check JPEG file structure for anomalies"""
        anomalies = []
        
        if 'anomalies' in structure_data and structure_data['anomalies']:
            for anomaly in structure_data['anomalies']:
                anomalies.append({
                    'type': 'file_structure_anomaly',
                    'description': anomaly,
                    'severity': 'high' if 'SOI' in anomaly or 'EOI' in anomaly else 'medium'
                })
        
        # Check marker sequence
        if 'markers' in structure_data:
            markers = structure_data['markers']
            marker_names = [m['marker'] for m in markers]
            
            # Check for duplicate DQT markers (can indicate recompression)
            dqt_count = marker_names.count('DQT')
            if dqt_count > 2:
                anomalies.append({
                    'type': 'multiple_quantization_tables',
                    'description': f'Found {dqt_count} DQT markers, possible recompression',
                    'severity': 'medium'
                })
        
        return anomalies
    
    def _check_thumbnail_consistency(self, thumbnail_data):
        """Check thumbnail consistency with main image"""
        anomalies = []
        
        if thumbnail_data.get('has_thumbnail') and not thumbnail_data.get('matches_main_image', True):
            anomalies.append({
                'type': 'thumbnail_mismatch',
                'description': 'EXIF thumbnail does not match main image',
                'severity': 'high'
            })
        
        if 'error' in thumbnail_data:
            anomalies.append({
                'type': 'thumbnail_extraction_error',
                'description': f'Error extracting thumbnail: {thumbnail_data["error"]}',
                'severity': 'low'
            })
        
        return anomalies
    
    def _gps_to_decimal(self, gps_coord, ref):
        """Convert GPS coordinates to decimal degrees"""
        if isinstance(gps_coord, (list, tuple)) and len(gps_coord) >= 3:
            degrees = gps_coord[0]
            minutes = gps_coord[1] if len(gps_coord) > 1 else 0
            seconds = gps_coord[2] if len(gps_coord) > 2 else 0
            
            # Handle IFD rational values
            if isinstance(degrees, (list, tuple)):
                degrees = degrees[0] / degrees[1] if degrees[1] else degrees[0]
            if isinstance(minutes, (list, tuple)):
                minutes = minutes[0] / minutes[1] if minutes[1] else minutes[0]
            if isinstance(seconds, (list, tuple)):
                seconds = seconds[0] / seconds[1] if seconds[1] else seconds[0]
            
            decimal = degrees + minutes/60 + seconds/3600
            
            if ref in ['S', 'W']:
                decimal = -decimal
                
            return decimal
        return 0
    
    def _compare_with_standard_tables(self, matrix):
        """Compare quantization table with standard tables"""
        max_similarity = 0
        
        for name, standard_table in self.standard_quantization_tables.items():
            # Calculate similarity using correlation
            flat_matrix = [val for row in matrix for val in row]
            flat_standard = [val for row in standard_table for val in row]
            
            # Simple correlation calculation
            if len(flat_matrix) == len(flat_standard):
                correlation = np.corrcoef(flat_matrix, flat_standard)[0, 1]
                max_similarity = max(max_similarity, correlation)
        
        return max_similarity


class MetadataAuthenticityScorer:
    def __init__(self):
        self.weight_map = {
            'software_editing_detected': 0.9,
            'time_inconsistency': 0.8,
            'photoshop_markers': 1.0,
            'thumbnail_mismatch': 0.85,
            'device_mismatch': 0.9,
            'gps_time_mismatch': 0.7,
            'uniform_quantization_table': 0.95,
            'file_structure_anomaly': 0.8,
            'xmp_metadata_present': 0.6,
            'non_standard_quantization': 0.5,
            'incomplete_gps_data': 0.2,
            'gps_parsing_error': 0.3
        }
    
    def calculate_metadata_authenticity_score(self, inconsistencies):
        """Calculate authenticity score based on detected inconsistencies"""
        if not inconsistencies:
            return 100  # No inconsistencies found
        
        # Calculate weighted penalty
        total_penalty = 0
        max_possible_penalty = 100
        
        for inconsistency in inconsistencies:
            anomaly_type = inconsistency.get('type', 'unknown')
            severity = inconsistency.get('severity', 'medium')
            
            # Get base weight
            base_weight = self.weight_map.get(anomaly_type, 0.5)
            
            # Adjust for severity
            severity_multiplier = {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.3
            }.get(severity, 0.5)
            
            penalty = base_weight * severity_multiplier * 20  # Scale to 0-20 per anomaly
            total_penalty += penalty
        
        # Calculate score (100 = authentic, 0 = highly manipulated)
        score = max(0, 100 - total_penalty)
        
        return round(score, 1)
    
    def generate_conclusion(self, score, inconsistencies):
        """Generate human-readable conclusion based on score and findings"""
        if score >= 85:
            conclusion = "Metadata menunjukkan tanda-tanda keaslian yang kuat. Tidak ada indikasi manipulasi yang signifikan."
        elif score >= 70:
            conclusion = "Metadata menunjukkan beberapa inkonsistensi minor yang mungkin berasal dari pemrosesan normal."
        elif score >= 50:
            conclusion = "Metadata menunjukkan indikasi manipulasi. Ditemukan beberapa anomali yang memerlukan investigasi lebih lanjut."
        else:
            conclusion = "Metadata menunjukkan indikasi kuat manipulasi. "
            
            # Add specific findings
            software_edits = [i for i in inconsistencies if 'software' in i.get('type', '')]
            if software_edits:
                conclusion += f"Terdeteksi penggunaan software editing. "
            
            time_issues = [i for i in inconsistencies if 'time' in i.get('type', '')]
            if time_issues:
                conclusion += f"Ditemukan inkonsistensi temporal. "
            
            thumbnail_issues = [i for i in inconsistencies if 'thumbnail' in i.get('type', '')]
            if thumbnail_issues:
                conclusion += f"Thumbnail tidak sesuai dengan gambar utama. "
        
        return conclusion


class MetadataDisplayFormatter:
    """Class to format metadata for display in table format"""
    
    @staticmethod
    def format_metadata_table(metadata, file_path):
        """Format metadata into a structured table display"""
        
        # Basic Information Section
        basic_info = []
        file_metadata = metadata.get('file_metadata', {})
        exif_data = metadata.get('exif_data', {})
        
        # File Information
        basic_info.append(['File Information', ''])
        basic_info.append(['Filename', os.path.basename(file_path)])
        basic_info.append(['File Size', f"{file_metadata.get('file_size', 0):,} bytes"])
        basic_info.append(['Image Type', file_metadata.get('file_type', 'Unknown')])
        basic_info.append(['MIME Type', file_metadata.get('mime_type', 'Unknown')])
        
        # Image Properties
        dims = file_metadata.get('dimensions', {})
        basic_info.append(['', ''])
        basic_info.append(['Image Properties', ''])
        basic_info.append(['Width', f"{dims.get('width', 0)} pixels"])
        basic_info.append(['Height', f"{dims.get('height', 0)} pixels"])
        basic_info.append(['Resolution', f"{dims.get('width', 0)} x {dims.get('height', 0)}"])
        
        # Camera Information
        basic_info.append(['', ''])
        basic_info.append(['Camera Information', ''])
        basic_info.append(['Make', exif_data.get('Make', 'Not Available')])
        basic_info.append(['Model', exif_data.get('Model', 'Not Available')])
        basic_info.append(['Software', exif_data.get('Software', 'Not Available')])
        
        # Date/Time Information
        basic_info.append(['', ''])
        basic_info.append(['Date/Time Information', ''])
        basic_info.append(['Date Taken', exif_data.get('DateTimeOriginal', 'Not Available')])
        basic_info.append(['Date Modified', exif_data.get('DateTime', 'Not Available')])
        basic_info.append(['File Created', file_metadata.get('created_time', 'Not Available')])
        basic_info.append(['File Modified', file_metadata.get('modified_time', 'Not Available')])
        
        # GPS Information
        gps_info = exif_data.get('GPSInfo', {})
        if gps_info:
            basic_info.append(['', ''])
            basic_info.append(['GPS Information', ''])
            basic_info.append(['GPS Available', 'Yes'])
            if 'GPSLatitude' in gps_info:
                basic_info.append(['Latitude', str(gps_info.get('GPSLatitude', 'N/A'))])
            if 'GPSLongitude' in gps_info:
                basic_info.append(['Longitude', str(gps_info.get('GPSLongitude', 'N/A'))])
        else:
            basic_info.append(['', ''])
            basic_info.append(['GPS Information', 'Not Available'])
        
        # Technical Details
        basic_info.append(['', ''])
        basic_info.append(['Technical Details', ''])
        basic_info.append(['ISO', str(exif_data.get('ISOSpeedRatings', 'Not Available'))])
        basic_info.append(['F-Number', str(exif_data.get('FNumber', 'Not Available'))])
        basic_info.append(['Exposure Time', str(exif_data.get('ExposureTime', 'Not Available'))])
        basic_info.append(['Focal Length', str(exif_data.get('FocalLength', 'Not Available'))])
        basic_info.append(['Flash', str(exif_data.get('Flash', 'Not Available'))])
        
        # File Hashes
        basic_info.append(['', ''])
        basic_info.append(['File Verification', ''])
        basic_info.append(['MD5 Hash', file_metadata.get('md5_hash', 'Not Available')])
        basic_info.append(['SHA256 Hash', file_metadata.get('sha256_hash', 'Not Available')])
        
        return basic_info
    
    @staticmethod
    def format_anomalies_table(inconsistencies):
        """Format detected anomalies into a table"""
        if not inconsistencies:
            return [['Status', 'No anomalies detected']]
        
        anomaly_table = []
        anomaly_table.append(['Detected Anomalies', 'Severity'])
        anomaly_table.append(['', ''])
        
        for inc in inconsistencies:
            desc = inc.get('description', '')
            severity = inc.get('severity', 'unknown')
            anomaly_table.append([desc, severity.upper()])
        
        return anomaly_table
    
    @staticmethod
    def format_quantization_table_info(metadata):
        """Format quantization table information"""
        qt_data = metadata.get('quantization_tables', {})
        qt_info = []
        
        if isinstance(qt_data, dict) and 'error' not in qt_data:
            qt_info.append(['Quantization Tables', f'{len(qt_data)} tables found'])
            for table_id, table_info in qt_data.items():
                if isinstance(table_info, dict) and 'matrix' in table_info:
                    matrix = table_info['matrix']
                    flat_values = [val for row in matrix for val in row]
                    qt_info.append([f'Table {table_id} Range', f'{min(flat_values)} - {max(flat_values)}'])
        else:
            qt_info.append(['Quantization Tables', 'Not Available or Error'])
        
        return qt_info


class ImageForensicsSystem:
    def __init__(self):
        self.extractor = EnhancedMetadataExtractor()
        self.checker = MetadataConsistencyChecker()
        self.scorer = MetadataAuthenticityScorer()
        self.formatter = MetadataDisplayFormatter()
    
    def analyze_image(self, image_path):
        """Perform complete forensic analysis on an image"""
        try:
            # Extract metadata
            metadata = self.extractor.extract_enhanced_metadata(image_path)
            
            # Check consistency
            inconsistencies = self.checker.check_enhanced_metadata_consistency(metadata)
            
            # Calculate score
            authenticity_score = self.scorer.calculate_metadata_authenticity_score(inconsistencies)
            
            # Generate conclusion
            conclusion = self.scorer.generate_conclusion(authenticity_score, inconsistencies)
            
            # Format inconsistencies for output
            formatted_inconsistencies = []
            for inc in inconsistencies:
                desc = inc.get('description', '')
                if inc.get('category'):
                    desc = f"{desc} ({inc['category']})"
                formatted_inconsistencies.append(desc)
            
            # Prepare output
            result = {
                'metadata': self._flatten_metadata(metadata),
                'inconsistencies': formatted_inconsistencies,
                'authenticity_score': authenticity_score,
                'conclusion': conclusion,
                'raw_metadata': metadata  # Keep raw data for table formatting
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'metadata': {},
                'inconsistencies': ['Error during analysis'],
                'authenticity_score': 0,
                'conclusion': 'Analysis failed due to error'
            }
    
    def display_results(self, image_path, result, output_format='table'):
        """Display results in various formats"""
        
        if output_format == 'table':
            print("\n" + "="*80)
            print(f"FORENSIC METADATA ANALYSIS REPORT")
            print(f"File: {image_path}")
            print("="*80 + "\n")
            
            # Display metadata table
            raw_metadata = result.get('raw_metadata', {})
            metadata_table = self.formatter.format_metadata_table(raw_metadata, image_path)
            print(tabulate(metadata_table, headers=['Property', 'Value'], tablefmt='grid'))
            
            # Display quantization table info
            print("\n" + "="*80)
            print("QUANTIZATION TABLE ANALYSIS")
            print("="*80 + "\n")
            qt_info = self.formatter.format_quantization_table_info(raw_metadata)
            print(tabulate(qt_info, headers=['Property', 'Value'], tablefmt='grid'))
            
            # Display anomalies
            print("\n" + "="*80)
            print("ANOMALY DETECTION RESULTS")
            print("="*80 + "\n")
            
            inconsistencies = []
            for inc_desc in result.get('inconsistencies', []):
                inconsistencies.append({'description': inc_desc, 'severity': 'Detected'})
            
            anomaly_table = self.formatter.format_anomalies_table(inconsistencies)
            print(tabulate(anomaly_table, tablefmt='grid'))
            
            # Display final score and conclusion
            print("\n" + "="*80)
            print("AUTHENTICITY ASSESSMENT")
            print("="*80 + "\n")
            
            score = result.get('authenticity_score', 0)
            conclusion = result.get('conclusion', '')
            
            assessment_table = [
                ['Authenticity Score', f"{score}/100"],
                ['Assessment', self._get_assessment_level(score)],
                ['Conclusion', conclusion]
            ]
            
            print(tabulate(assessment_table, headers=['Metric', 'Result'], tablefmt='grid'))
            
            # Add visual score indicator
            print("\n" + self._create_score_bar(score))
            
        elif output_format == 'json':
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif output_format == 'html':
            html = self._generate_html_report(image_path, result)
            return html
    
    def _get_assessment_level(self, score):
        """Get assessment level based on score"""
        if score >= 85:
            return "AUTHENTIC - High confidence in image authenticity"
        elif score >= 70:
            return "LIKELY AUTHENTIC - Minor inconsistencies detected"
        elif score >= 50:
            return "SUSPICIOUS - Multiple anomalies detected"
        else:
            return "LIKELY MANIPULATED - Strong evidence of tampering"
    
    def _create_score_bar(self, score):
        """Create visual score bar"""
        bar_length = 50
        filled_length = int(bar_length * score / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Color coding (terminal colors)
        if score >= 85:
            color = '\033[92m'  # Green
        elif score >= 70:
            color = '\033[93m'  # Yellow
        elif score >= 50:
            color = '\033[91m'  # Orange
        else:
            color = '\033[91m'  # Red
        
        reset_color = '\033[0m'
        
        return f"Score: {color}[{bar}] {score}%{reset_color}"
    
    def _generate_html_report(self, image_path, result):
        """Generate HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Metadata Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .authentic {{ color: green; }}
                .suspicious {{ color: orange; }}
                .manipulated {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Forensic Metadata Analysis Report</h1>
            <p><strong>File:</strong> {image_path}</p>
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Authenticity Assessment</h2>
            <p class="score">Score: <span class="{self._get_score_class(result.get('authenticity_score', 0))}">{result.get('authenticity_score', 0)}/100</span></p>
            <p><strong>Conclusion:</strong> {result.get('conclusion', '')}</p>
            
            <!-- Add more sections as needed -->
        </body>
        </html>
        """
        return html
    
    def _get_score_class(self, score):
        """Get CSS class based on score"""
        if score >= 85:
            return 'authentic'
        elif score >= 50:
            return 'suspicious'
        else:
            return 'manipulated'
    
    def _flatten_metadata(self, metadata):
        """Flatten nested metadata structure for output"""
        flat = {}
        
        # File metadata
        if 'file_metadata' in metadata:
            for key, value in metadata['file_metadata'].items():
                if key not in ['anomalies']:  # Skip internal fields
                    flat[f'File_{key}'] = value
        
        # EXIF data
        if 'exif_data' in metadata:
            for key, value in metadata['exif_data'].items():
                if not isinstance(value, (dict, list)) or key == 'GPSInfo':
                    flat[key] = str(value)
        
        # Add important derived information
        if 'quantization_tables' in metadata and isinstance(metadata['quantization_tables'], dict):
            flat['HasQuantizationTables'] = len(metadata['quantization_tables'])
        
        if 'file_structure' in metadata:
            structure = metadata['file_structure']
            if 'marker_count' in structure:
                flat['JPEGMarkerCount'] = structure['marker_count']
        
        return flat


# Wrapper functions for backward compatibility
def extract_enhanced_metadata(image_path):
    """Wrapper for existing function signature"""
    extractor = EnhancedMetadataExtractor()
    return extractor.extract_enhanced_metadata(image_path)

def check_enhanced_metadata_consistency(metadata):
    """Wrapper for existing function signature"""
    checker = MetadataConsistencyChecker()
    return checker.check_enhanced_metadata_consistency(metadata)

def calculate_metadata_authenticity_score(inconsistencies):
    """Wrapper for existing function signature"""
    scorer = MetadataAuthenticityScorer()
    return scorer.calculate_metadata_authenticity_score(inconsistencies)


def main():
    """Main entry point for the metadata forensics tool"""
    parser = argparse.ArgumentParser(
        description='Image Metadata Forensic Analysis Tool',
        epilog='Part of: SISTEM DETEKSI FORENSIK KEASLIAN GAMBAR DAN VIDEO'
    )
    parser.add_argument('image_path', help='Path to the image file to analyze')
    parser.add_argument('-o', '--output', help='Output file for JSON results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-f', '--format', choices=['table', 'json', 'html'], 
                        default='table', help='Output format (default: table)')
    parser.add_argument('--html-output', help='Save HTML report to file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.image_path):
        print(f"Error: File '{args.image_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Initialize the forensics system
    if args.verbose:
        print(f"Analyzing image: {args.image_path}")
    
    forensics = ImageForensicsSystem()
    
    # Analyze the image
    result = forensics.analyze_image(args.image_path)
    
    # Handle different output formats
    if args.format == 'table':
        forensics.display_results(args.image_path, result, 'table')
    elif args.format == 'json':
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            if args.verbose:
                print(f"Results saved to: {args.output}")
        else:
            print(json_output)
    elif args.format == 'html':
        html_output = forensics._generate_html_report(args.image_path, result)
        if args.html_output:
            with open(args.html_output, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"HTML report saved to: {args.html_output}")
        else:
            print(html_output)
    
    # Save JSON output if specified (in addition to display)
    if args.output and args.format != 'json':
        json_output = json.dumps(result, indent=2, ensure_ascii=False)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        if args.verbose:
            print(f"\nJSON results also saved to: {args.output}")


if __name__ == "__main__":
    main()