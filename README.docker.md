Docker usage

Build the image:

```bash
docker build -t advanced-doc-rag:latest .
```

Run the container (exposes port 8000):

```bash
docker run -p 8000:8000 advanced-doc-rag:latest
```

Notes:

- The Dockerfile installs `libmagic1`, `poppler-utils`, and `tesseract-ocr` needed by `unstructured[pdf]`.
- The Dockerfile installs Python packages from `requirements.txt`. `unstructured[pdf]` is included in `requirements.txt`.
- If you prefer all document-type dependencies, add `unstructured[all-docs]` to `requirements.txt` instead of `unstructured[pdf]`.
