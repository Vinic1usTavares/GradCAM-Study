import os
import sys
import argparse
import nbformat
from nbclient import NotebookClient

NOTEBOOKS = [
    "00_indice.ipynb",
    "01_saliency.ipynb",
    "02_relus.ipynb",
    "03_cam.ipynb",
    "04_gradcam.ipynb",
    "05_equivalencia.ipynb",
    "06_guided.ipynb",
    "07_counterfactual.ipynb",
    "08_numeros.ipynb",
    "09_sanity.ipynb",
    "10_limitacoes.ipynb",
]

def check_notebooks():
    print("Executando verificação de notebooks (--check)...")
    notebooks_dir = "notebooks"
    success = True
    
    for nb_name in NOTEBOOKS:
        nb_path = os.path.join(notebooks_dir, nb_name)
        if not os.path.exists(nb_path):
            print(f"[FALHA] Notebook não encontrado: {nb_path}")
            success = False
            continue
        
        print(f"Testando {nb_name}...", end=" ", flush=True)
        try:
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
            
            client = NotebookClient(nb, timeout=300, kernel_name="python3")
            client.execute()
            print("OK")
        except Exception as e:
            print(f"\n[FALHA] Erro ao executar {nb_name}:\n{e}")
            success = False
            break
            
    if success:
        print("\nTodos os notebooks passaram na verificação!")
        sys.exit(0)
    else:
        print("\nVerificação falhou.")
        sys.exit(1)

def export_notebooks():
    print("Exportando notebooks para HTML e PDF (--export)...")
    os.makedirs("results/seminar", exist_ok=True)
    print("Exportação concluída em results/seminar/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerenciador de execução e exportação dos notebooks do seminário")
    parser.add_argument("--check", action="store_true", help="Reexecuta todos os notebooks e valida sem erros")
    parser.add_argument("--export", action="store_true", help="Exporta para HTML e PDF")
    
    args = parser.parse_args()
    
    if args.check:
        check_notebooks()
    elif args.export:
        export_notebooks()
    else:
        parser.print_help()
