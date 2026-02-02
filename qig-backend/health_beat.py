from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'celery-beat', 'version': '1.0.0'}), 200

@app.route('/health', methods=['GET'])
def health_alt():
    return jsonify({'status': 'healthy', 'service': 'celery-beat', 'version': '1.0.0'}), 200

def run_health_server():
    port = int(os.getenv('FLASK_PORT') or os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_health_server()
