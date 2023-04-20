import http.server
import socketserver
from Predict_next_week_spending import run_script
from predict_spender_script import run_spender_script
import os
PORT = os.getenv('PORT') or 8080

class MyRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        message = "Hello, World!"
        self.wfile.write(bytes(message, "utf8"))
        print("Running scripts!!!!!")
        run_script()
        run_spender_script()

Handler = MyRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Server started at localhost:{}".format(PORT))
    httpd.serve_forever()