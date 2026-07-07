"""
train_segmentation.py
=====================
Script para fine-tuning de um modelo YOLO para segmentação de instâncias
(ex: caixas de papelão).

Uso rápido:
    python train_segmentation.py                        # treina com configurações padrão
    python train_segmentation.py --yaml caminho/data.yaml
    python train_segmentation.py --model yolo26n-seg.pt --epochs 150 --batch 128
    python train_segmentation.py --predict --source imagem.jpg
    python train_segmentation.py --export                # copia best.pt para Downloads
"""

import argparse
import random
import shutil
from pathlib import Path

DEFAULT_MODEL  = "yolo26n-seg.pt"  
DEFAULT_YAML   = r"Dataset/data.yaml"
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 128                  # reduza para 4 se tiver pouca VRAM
DEFAULT_DEVICE = 0                  # 0 = primeira GPU; "cpu" para CPU
DEFAULT_WORKERS = 2

def limpar_labels_mistos(label_dir: str, dry_run: bool = False) -> int:
    """
    Percorre todos os .txt em label_dir (recursivo) e remove linhas com exatamente
    5 campos (formato detect) quando o arquivo também contém linhas de segmentação.

    Args:
        label_dir: Caminho para a pasta de labels (ex: 'Dataset/train/labels').
        dry_run:   Se True, apenas reporta sem modificar nenhum arquivo.

    Returns:
        Número de arquivos modificados (ou que seriam modificados em dry_run).
    """
    label_dir = Path(label_dir)
    if not label_dir.exists():
        print(f"Pasta de labels não encontrada: {label_dir} — pulando.")
        return 0

    modificados = 0
    for txt in sorted(label_dir.rglob("*.txt")):
        linhas = txt.read_text(encoding="utf-8").splitlines()
        linhas_validas = [l for l in linhas if l.strip()]  # ignora linhas vazias

        tem_seg    = any(len(l.split()) > 5  for l in linhas_validas)
        tem_detect = any(len(l.split()) == 5 for l in linhas_validas)

        if tem_seg and tem_detect:
            linhas_limpas = [l for l in linhas if not (l.strip() and len(l.split()) == 5)]
            if not dry_run:
                txt.write_text("\n".join(linhas_limpas), encoding="utf-8")
            modificados += 1
            removidas = len(linhas_validas) - len([l for l in linhas_limpas if l.strip()])
            flag = "[DRY-RUN] " if dry_run else ""
            print(f"  {flag}{txt.name}: {removidas} linha(s) detect removida(s)")

    return modificados


def limpar_dataset(yaml_path: str, dry_run: bool = False):
    """Aplica limpar_labels_mistos nos splits train e valid do dataset."""
    import yaml as _yaml

    with open(yaml_path, "r") as f:
        cfg = _yaml.safe_load(f)

    dataset_root = Path(yaml_path).parent
    splits = ["train", "val", "valid", "test"]

    print(f"\n{'─'*60}")
    print(f"  Limpando labels mistos em: {dataset_root}")
    if dry_run:
        print("  (modo dry-run — nenhum arquivo será alterado)")
    print(f"{'─'*60}")

    caches_deletados = 0
    for cache in dataset_root.rglob("labels.cache"):
        if not dry_run:
            cache.unlink()
        flag = "[DRY-RUN] " if dry_run else ""
        print(f"\n  {flag}Cache removido: {cache}")
        caches_deletados += 1
    if caches_deletados == 0:
        print("\n  Nenhum labels.cache encontrado.")

    total = 0
    for split in splits:
        # Tenta descobrir o caminho do split via yaml ou convenção
        img_dir_rel = cfg.get(split, f"{split}/images")
        img_dir = dataset_root / img_dir_rel
        label_dir = img_dir.parent / "labels" if img_dir.exists() else dataset_root / split / "labels"
        if label_dir.exists():
            print(f"\n  [{split}] → {label_dir}")
            n = limpar_labels_mistos(label_dir, dry_run=dry_run)
            total += n
        else:
            # Tenta variação direta
            alt = dataset_root / split / "labels"
            if alt.exists():
                print(f"\n  [{split}] → {alt}")
                n = limpar_labels_mistos(alt, dry_run=dry_run)
                total += n

    print(f"\n{'─'*60}")
    acao = "seriam modificados" if dry_run else "modificados"
    print(f"  Total de arquivos {acao}: {total}")
    print(f"{'─'*60}\n")
    return total


def treinar(yaml_path: str, model_name: str, epochs: int, imgsz: int,
            batch: int, device, workers: int):
    from ultralytics import YOLO

    if not Path(yaml_path).exists():
        raise FileNotFoundError(
            f"Arquivo data.yaml não encontrado em: {yaml_path}\n"
            "Ajuste DEFAULT_YAML ou passe --yaml com o caminho correto."
        )

    # Limpa labels mistos (detect + segment) antes de treinar
    limpar_dataset(yaml_path)

    print(f"\n{'='*60}")
    print(f"  Iniciando treinamento de SEGMENTAÇÃO DE INSTÂNCIAS")
    print(f"  Modelo base : {model_name}")
    print(f"  Dataset     : {yaml_path}")
    print(f"  Épocas      : {epochs}  |  imgsz: {imgsz}  |  batch: {batch}")
    print(f"  Device      : {device}")
    print(f"{'='*60}\n")

    model = YOLO(model_name)

    results = model.train(
        data=yaml_path,
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        plots=True,
    )

    pasta_treino = Path(results.save_dir)
    best_pt = pasta_treino / "weights" / "best.pt"

    print(f"\nTreinamento concluído.")
    print(f"  Resultados salvos em: {pasta_treino}")
    if best_pt.exists():
        print(f"  Melhores pesos      : {best_pt}")

    return results


def predizer(source: str, model_path: str = None):
    from ultralytics import YOLO

    # Se não informar um modelo treinado, tenta encontrar o mais recente
    if model_path is None:
        model_path = _encontrar_melhor_peso()

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    print(f"\nRodando inferência com: {model_path}")
    print(f"  Fonte: {source}\n")

    model = YOLO(model_path)
    results = model.predict(source=source, task="segment", save=True, verbose=True)

    print(f"\nResultados salvos em: {Path(results[0].save_dir)}")
    return results

def visualizar(yaml_path: str, model_path: str = None):
    """Roda inferência em uma imagem aleatória do split 'test' do dataset."""
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    from ultralytics import YOLO
    import yaml

    if model_path is None:
        model_path = _encontrar_melhor_peso()

    # Lê o yaml para descobrir onde ficam as imagens de teste
    with open(yaml_path, "r") as f:
        dataset_cfg = yaml.safe_load(f)

    dataset_root = Path(yaml_path).parent
    test_dir = dataset_root / dataset_cfg.get("test", "test/images")

    if not test_dir.exists():
        # Tenta variações comuns
        for candidato in ["test/images", "valid/images", "val/images"]:
            if (dataset_root / candidato).exists():
                test_dir = dataset_root / candidato
                break
        else:
            raise FileNotFoundError(
                f"Pasta de imagens de teste não encontrada. "
                f"Verifique o campo 'test' no seu data.yaml."
            )

    imagens = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not imagens:
        raise ValueError(f"Nenhuma imagem encontrada em {test_dir}")

    img_path = random.choice(imagens)
    print(f"\nTestando com imagem: {img_path.name}")

    model = YOLO(model_path)
    result = model.predict(PILImage.open(img_path), task="segment", verbose=False)[0]

    # Plota o resultado com as máscaras
    annotated = result.plot()  # retorna numpy array com anotações

    plt.figure(figsize=(10, 10))
    plt.imshow(annotated[..., ::-1])  # BGR → RGB
    plt.title(f"Segmentação: {img_path.name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def exportar_pesos(diretorio_destino: str = None):
    """Copia o best.pt do treino mais recente para uma pasta de fácil acesso."""
    best_pt = _encontrar_melhor_peso(retornar_path=True)

    if diretorio_destino is None:
        diretorio_destino = Path.home() / "Downloads"
    else:
        diretorio_destino = Path(diretorio_destino)

    diretorio_destino.mkdir(parents=True, exist_ok=True)

    # Nome com sufixo da pasta de treino para não sobrescrever versões anteriores
    nome_final = f"yolo_seg_best_{best_pt.parent.parent.name}.pt"
    caminho_final = diretorio_destino / nome_final

    shutil.copy(best_pt, caminho_final)
    print("=" * 54)
    print("Pesos exportados.")
    print(f"  Origem : {best_pt}")
    print(f"  Destino: {caminho_final.resolve()}")
    print("=" * 54)

def _encontrar_melhor_peso(retornar_path: bool = False):
    # YOLO salva treinos de segmentação em runs/segment/train*
    runs_dir = Path("runs/segment")
    if not runs_dir.exists():
        raise FileNotFoundError(
            "Pasta 'runs/segment' não encontrada. "
            "Execute o treinamento primeiro (--train)."
        )

    pastas = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
    if not pastas:
        raise FileNotFoundError("Nenhuma pasta de treino encontrada em runs/segment.")

    def _sufixo(p):
        nome = p.name
        return 0 if nome == "train" else int(nome.replace("train", "") or 0)

    pasta_recente = max(pastas, key=_sufixo)
    best_pt = pasta_recente / "weights" / "best.pt"

    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt não encontrado em {best_pt}")

    if retornar_path:
        return best_pt
    return str(best_pt)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning YOLO para segmentação de instâncias."
    )
    parser.add_argument("--train",   action="store_true", help="Executa o treinamento (padrão se nada for passado)")
    parser.add_argument("--predict", action="store_true", help="Roda inferência numa imagem")
    parser.add_argument("--visualize", action="store_true", help="Visualiza resultado em imagem aleatória do test set")
    parser.add_argument("--export",  action="store_true", help="Copia best.pt para a pasta Downloads")
    parser.add_argument("--clean",   action="store_true", help="Remove linhas detect de labels mistos (roda automaticamente antes do treino)")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Usa com --clean: mostra o que seria alterado sem modificar arquivos")

    # Parâmetros de treino
    parser.add_argument("--yaml",    default=DEFAULT_YAML,   help=f"Caminho do data.yaml (padrão: {DEFAULT_YAML})")
    parser.add_argument("--model",   default=DEFAULT_MODEL,  help=f"Modelo base (padrão: {DEFAULT_MODEL})")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS,  help=f"Épocas (padrão: {DEFAULT_EPOCHS})")
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ,   help=f"Tamanho da imagem (padrão: {DEFAULT_IMGSZ})")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH,   help=f"Batch size (padrão: {DEFAULT_BATCH})")
    parser.add_argument("--device",  default=DEFAULT_DEVICE, help="Device: 0 para GPU, 'cpu' para CPU")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Workers (padrão: {DEFAULT_WORKERS})")

    # Parâmetros de inferência
    parser.add_argument("--source",  default=None, help="Imagem/pasta/vídeo para inferência (--predict)")
    parser.add_argument("--weights", default=None, help="Caminho do .pt para inferência (opcional)")
    parser.add_argument("--dest",    default=None, help="Destino para exportar pesos (padrão: ~/Downloads)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Se nenhuma ação for explicitada, assume --train
    nenhuma_acao = not any([args.train, args.predict, args.visualize, args.export, args.clean])
    if nenhuma_acao:
        args.train = True

    if args.clean:
        limpar_dataset(args.yaml, dry_run=args.dry_run)

    if args.train:
        treinar(
            yaml_path=args.yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
        )

    if args.predict:
        if args.source is None:
            print("--predict requer --source <imagem ou pasta>")
        else:
            predizer(source=args.source, model_path=args.weights)

    if args.visualize:
        visualizar(yaml_path=args.yaml, model_path=args.weights)

    if args.export:
        exportar_pesos(diretorio_destino=args.dest)


if __name__ == "__main__":
    main()