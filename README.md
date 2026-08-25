# REHAB

## Estandarizado para GitHub

Este repositorio incluye archivos estándar para facilitar colaboración en GitHub:

- [`.gitignore`](.gitignore): Ignora entornos virtuales, caches y checkpoints de Jupyter.
- [`.gitattributes`](.gitattributes): Mejora el manejo de notebooks en git.
- [`LICENSE`](LICENSE): Licencia MIT por defecto.
- [`requirements.txt`](requirements.txt): Dependencias para ejecutar el notebook.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml): Workflow que ejecuta el notebook en CI.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): Guía rápida para contribuir.

Instrucciones rápidas para reproducir localmente:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Para preparar el repositorio antes de subir a GitHub:

```bash
git init
git add .
git commit -m "Initial standardization for GitHub"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
