"""
Generates a self-signed TLS certificate + private key for running the
backend over HTTPS on an isolated local network — closing the "real
security beyond the API key" gap flagged in earlier rounds (plain HTTP
means the API key itself travels in the clear).

This is deliberately NOT a CA-signed certificate — there's no certificate
authority reachable from an offline standalone system, and that's fine:
browsers will show a one-time "not trusted" warning to click through
(expected for a self-signed cert on a private network), but the
connection itself is still genuinely encrypted. This is the right and
common approach for local/offline HTTPS, not a security shortcut.

Usage:
    python scripts/generate_self_signed_cert.py [--output-dir certs] [--days 825] [--ip 192.168.1.50]

Then run the backend with TLS:
    uvicorn backend.main:app --port 8000 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem
"""
import argparse
import ipaddress
import os
import socket
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="certs")
    parser.add_argument("--days", type=int, default=825, help="Validity period (825 is the max most browsers accept for self-signed certs)")
    parser.add_argument("--ip", action="append", default=[],
                         help="Additional IP address(es) to include (repeatable) — needed if you'll access "
                              "the backend by LAN IP, not just localhost. e.g. --ip 192.168.1.50")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    key_path = os.path.join(args.output_dir, "key.pem")
    cert_path = os.path.join(args.output_dir, "cert.pem")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "ASTRA Local"),
    ])

    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        # Best-effort: include this machine's own LAN IP automatically so
        # the common case (accessing via LAN IP from another device) works
        # without the operator needing to know it upfront.
        hostname_ip = socket.gethostbyname(socket.gethostname())
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(hostname_ip)))
    except Exception:
        pass   # not fatal — --ip can supply it explicitly instead
    for ip_str in args.ip:
        try:
            san_entries.append(x509.IPAddress(ipaddress.IPv4Address(ip_str)))
        except ValueError:
            print(f"WARNING: '{ip_str}' isn't a valid IPv4 address, skipping.")

    now = datetime.now(timezone.utc)
    # Dedupe (e.g. localhost's own IP and an explicit --ip can coincide)
    # while preserving order — cryptography's builder doesn't dedupe for you.
    seen = set()
    deduped_sans = []
    for entry in san_entries:
        entry_key = str(entry)
        if entry_key not in seen:
            seen.add(entry_key)
            deduped_sans.append(entry)
    san_entries = deduped_sans

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=args.days))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Saved private key to {key_path}")
    print(f"Saved certificate to {cert_path}")
    print(f"Valid for: {', '.join(str(e) for e in san_entries)}")
    print()
    print("Run the backend with TLS:")
    print(f"  uvicorn backend.main:app --port 8000 --ssl-keyfile {key_path} --ssl-certfile {cert_path}")
    print()
    print("Browsers will show a one-time 'not trusted' warning for the self-signed cert — click through it.")
    print("This is expected on an isolated local network with no reachable certificate authority; the")
    print("connection is still genuinely encrypted.")


if __name__ == "__main__":
    main()
