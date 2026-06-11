import os
import socket
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def main():
    port = CFG.get("vr_https_port", 8443)
    url = f"https://{lan_ip()}:{port}/"
    print(f"Open this in the VR headset browser: {url}")
    try:
        import qrcode
        qr = qrcode.QRCode()
        qr.add_data(url)
        qr.make()
        qr.print_ascii(invert=True)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
