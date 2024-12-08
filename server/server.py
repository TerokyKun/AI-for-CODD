from flask import Flask, request, jsonify, send_from_directory
import os
import subprocess
import shutil
from flask_cors import CORS
from werkzeug.utils import secure_filename
from main import run_video_analysis  # Импорт вашей функции анализа видео

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

UPLOAD_FOLDER = './uploads'
HLS_FOLDER = './hls'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HLS_FOLDER, exist_ok=True)

# Очередь для удаления файлов (сначала добавляем в очередь, потом удаляем)
files_to_delete = []

@app.route('/upload', methods=['POST'])
def upload_video():
    """Прием, анализ и обработка видеофайла."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    video_file = request.files['file']
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    hls_output_path = os.path.join(HLS_FOLDER, filename.split('.')[0])

    # Сохранение видео на сервер
    video_file.save(video_path)

    # Путь для выходного видео (например, временный файл в папке `uploads`)
    output_path = os.path.join(UPLOAD_FOLDER, f"processed_{filename}")

    # Запуск анализа видео с передачей пути для выходного видео
    try:
        analysis_results = run_video_analysis(video_path, output_path)
    except Exception as e:
        return jsonify({'error': f'Error analyzing video: {e}'}), 500

    # Преобразование обработанного видео в HLS с помощью FFmpeg
    try:
        os.makedirs(hls_output_path, exist_ok=True)
        command = [
            "ffmpeg", "-i", output_path,  # Используем выходное видео после анализа
            "-codec:v", "libx264", "-codec:a", "aac", "-strict", "experimental",
            "-start_number", "0", "-hls_time", "10", "-hls_list_size", "0",
            "-f", "hls", os.path.join(hls_output_path, "master.m3u8")
        ]
        subprocess.run(command, check=True)
        if not os.path.exists(os.path.join(hls_output_path, "master.m3u8")):
            return jsonify({'error': 'HLS playlist not created'}), 500
    except Exception as e:
        return jsonify({'error': f'Error processing video: {e}'}), 500

    # Добавление файлов в очередь на удаление
    files_to_delete.append({
        'video_path': video_path,
        'output_path': output_path,
        'hls_output_path': hls_output_path
    })

    return jsonify({
        'status': 'ready',
        'hls_path': f"{filename.split('.')[0]}/master.m3u8",
        'analysis': analysis_results
    }), 201

@app.route('/clean', methods=['GET'])
def clean_files():
    """Удаление файлов после получения запроса."""
    global files_to_delete
    try:
        # Процесс удаления файлов из очереди
        for file_info in files_to_delete:
            try:
                # Удаление видео и обработанных файлов
                os.remove(file_info['video_path'])
            except Exception as e:
                print(f"Error deleting original video: {e}")
                
            try:
                os.remove(file_info['output_path'])
            except Exception as e:
                print(f"Error deleting processed video: {e}")
                
            try:
                shutil.rmtree(file_info['hls_output_path'])
            except Exception as e:
                print(f"Error deleting HLS folder: {e}")
        
        # Очистка очереди после удаления файлов
        files_to_delete = []

        return jsonify({'status': 'cleaned', 'files_deleted': len(files_to_delete)}), 200
    except Exception as e:
        return jsonify({'error': f"Error cleaning files: {e}"}), 500


@app.route('/stream/<path:filename>', methods=['GET'])
def hls_stream(filename):
    return send_from_directory(HLS_FOLDER, filename)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ready'})

if __name__ == '__main__':
    app.run(debug=True)
