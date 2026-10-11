Place extra root CA certificates (PEM format, `*.crt`) here if your network uses a
TLS-intercepting proxy. They are trusted during the backend image build (pip) and at runtime.
Files in this folder other than this README and .gitkeep are git-ignored.

macOS example (exports the System keychain roots):

    security find-certificate -a -p /Library/Keychains/System.keychain > docker/certs/host-roots.crt
