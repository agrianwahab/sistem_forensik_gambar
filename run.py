import eventlet
eventlet.monkey_patch() # Panggil monkey_patch() di paling atas

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    # Gunakan socketio.run untuk server pengembangan Flask-SocketIO
    # Ini akan menggunakan server pengembangan Werkzeug dengan dukungan WebSocket (misalnya, eventlet atau gevent)
    # Pastikan eventlet atau gevent terinstal (lihat requirements.txt)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)