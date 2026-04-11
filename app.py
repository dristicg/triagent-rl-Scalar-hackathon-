import http.server
import socketserver
import os

# Port 7860 is the default for Hugging Face Spaces
PORT = 7860

Handler = http.server.SimpleHTTPRequestHandler

def main():
    print(f"Keeping Space awake on port {PORT}...")
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.shutdown()

if __name__ == "__main__":
    main()
