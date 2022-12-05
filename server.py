import socketio
from waitress import serve
from main import app
import face_recognition as faceRec

sio = socketio.Server()
appServer = socketio.WSGIApp(sio, app)



if __name__ == '__main__':
    serve(appServer, host='0.0.0.0', port=8080, url_scheme='http', threads=6)