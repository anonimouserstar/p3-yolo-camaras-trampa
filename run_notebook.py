"""Ejecuta el notebook completo de extremo a extremo (sin abrir Jupyter).

Uso:
    python run_notebook.py            # experimento completo (requiere GPU)
    python run_notebook.py --smoke    # prueba rápida con datos sintéticos (CPU, ~15 min)
"""
import argparse, os, sys, time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", default="P3_YOLO_CamarasTrampa.ipynb")
ap.add_argument("--output", default="P3_YOLO_CamarasTrampa.executed.ipynb")
ap.add_argument("--smoke", action="store_true", help="datos sintéticos, sin descargas")
ap.add_argument("--kernel", default="python3")
args = ap.parse_args()

if args.smoke:
    os.environ["P3_SMOKE_TEST"] = "1"
nb = nbformat.read(args.notebook, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name=args.kernel,
                        resources={"metadata": {"path": os.path.dirname(os.path.abspath(args.notebook))}})
t0 = time.time()
try:
    client.execute()
    print(f"Notebook ejecutado correctamente en {time.time() - t0:.0f} s -> {args.output}")
    code = 0
except CellExecutionError as e:
    print("Error durante la ejecución:\n", str(e)[-4000:])
    code = 1
finally:
    nbformat.write(nb, args.output)
sys.exit(code)
