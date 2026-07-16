import sys
import os
import time
import math
import random

import pandas as pd

# Kau'yon: el bucle itera con calma hasta que los centros dejan de moverse. Cota de seguridad por si nunca convergen.
KAUYON = 100
TOL = 1e-9


# ========================= CAPA LOGICA — K-MEANS =========================

def normalize_by_max(df, columns):
    """Regla de 3 con el maximo = 1 (metodo del documento): valor / max_columna."""
    fio = df.copy()  # Fio: prepara la materia prima
    for col in columns:
        col_max = pd.to_numeric(fio[col]).max()
        fio[col] = pd.to_numeric(fio[col]) / col_max if col_max else 0.0
    return fio


def euclidean(a, b):
    """Longitud del segmento entre dos puntos en N dimensiones."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def assign_clusters(points, centroids):
    """Mont'ka: cada punto va a su centro mas cercano."""
    montka = []
    for p in points:
        distances = [euclidean(p, ctr) for ctr in centroids]
        montka.append(distances.index(min(distances)))
    return montka


def update_centroids(points, labels, k, previous):
    """Nuevo centro = promedio de los puntos del cluster (o el previo si queda vacio)."""
    dims = len(points[0])
    centroids = []
    for cluster in range(k):
        members = [points[i] for i in range(len(points)) if labels[i] == cluster]
        if not members:
            centroids.append(previous[cluster])
            continue
        centroids.append([sum(m[d] for m in members) / len(members) for d in range(dims)])
    return centroids


def inertia(points, labels, centroids):
    """Tau'va: suma de distancias intra-cluster que se minimiza."""
    return sum(euclidean(points[i], centroids[labels[i]]) for i in range(len(points)))


def kmeans(points, k, init_centroids=None, seed=42):
    if init_centroids is None:
        random.seed(seed)
        init_centroids = [list(points[i]) for i in random.sample(range(len(points)), k)]
    centroids = [list(ctr) for ctr in init_centroids]
    history = []
    for step in range(KAUYON):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, k, centroids)
        tau_va = inertia(points, labels, new_centroids)
        history.append((step + 1, labels, new_centroids, tau_va))
        moved = max(euclidean(a, b) for a, b in zip(centroids, new_centroids))
        centroids = new_centroids
        if moved <= TOL:
            break
    return labels, centroids, history


# ====================== CAPA INTERFAZ — tema terminal ======================

def _supports_color():
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR")


_COLOR = True

_PAL = {
    "cyan": "38;5;51", "teal": "38;5;44", "green": "38;5;48",
    "dim": "38;5;245", "faint": "38;5;239", "amber": "38;5;214",
    "red": "38;5;203", "white": "38;5;231", "magenta": "38;5;177",
}


def c(text, color, bold=False):
    if not _COLOR:
        return str(text)
    return f"\033[{'1;' if bold else ''}{_PAL.get(color, '0')}m{text}\033[0m"


def tw(text="", speed=0.006, color=None, bold=False, end="\n"):
    if color and _COLOR:
        sys.stdout.write(("\033[1m" if bold else "") + f"\033[{_PAL.get(color, '0')}m")
    for char in str(text):
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    if color and _COLOR:
        sys.stdout.write("\033[0m")
    sys.stdout.write(end)
    sys.stdout.flush()


def out(text="", color=None, bold=False):
    print(c(text, color, bold) if color else text)


W = 62


def scanline():
    out(c("▓▒░" + "─" * (W - 6) + "░▒▓", "faint"))


def section(title, jp=""):
    print()
    label = c(title, "cyan", bold=True)
    visible = len(f"╭─╴ {title} {jp}")
    fill = max(0, W - visible - 1)
    out(f"{c('╭─╴', 'cyan')} {label} {c(jp, 'teal')} {c('╶' * fill + '╮', 'cyan')}")


def field(label, value):
    out(f"  {c(label, 'dim')} {c('│', 'faint')} {c(value, 'green')}")


def ok(msg):
    out(f"  {c('[', 'faint')}{c('OK', 'green', True)}{c(']', 'faint')} {c(msg, 'dim')}")


def info(msg):
    out(f"  {c('[', 'faint')}{c('··', 'cyan')}{c(']', 'faint')} {c(msg, 'dim')}")


def warn(msg):
    out(f"  {c('[', 'faint')}{c('!!', 'amber', True)}{c(']', 'faint')} {c(msg, 'amber')}")


def err(msg):
    out(f"  {c('[', 'faint')}{c('XX', 'red', True)}{c(']', 'faint')} {c(msg, 'red')}")


def ask(label):
    sys.stdout.write(c(f"  {label}\n", "amber") + c("  ╴ ", "cyan"))
    sys.stdout.flush()
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        print()
        die("SESSION TERMINATED BY OPERATOR")


def die(msg):
    print()
    err(msg)
    scanline()
    sys.exit(1)


def enable_ansi():
    if os.name == "nt":
        os.system("")


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    art = [
        "╔════════════════════════════════════════════════════════════╗",
        "║                                                            ║",
        "║      K - M E A N S   C L U S T E R I N G   E N G I N E     ║",
        "║       ───────────────────────────────────────────────      ║",
        "║        群化 // PROJECT 2501 // FOR THE GREATER GOOD        ║",
        "║                                                            ║",
        "╚════════════════════════════════════════════════════════════╝",
    ]
    print()
    for i, line in enumerate(art):
        tw(line, 0.0009, "cyan" if i in (2, 4) else "teal", bold=(i == 2))


def boot_sequence():
    steps = [
        "MOUNTING CORE MODULES", "LINKING DISTANCE-METRIC BUS",
        "LOADING K-MEANS KERNEL", "CALIBRATING CENTROID DRIVES",
        "ESTABLISHING OPERATOR UPLINK",
    ]
    print()
    for s in steps:
        sys.stdout.write(c(f"  ░ {s} ", "dim"))
        sys.stdout.flush()
        time.sleep(0.12)
        out(c("·" * (42 - len(s)), "faint") + " " + c("[OK]", "green", True))
    print()
    ok("ALL SYSTEMS NOMINAL // STANDING BY")


# ============================ FLUJO PRINCIPAL ============================

def load_dataset(path):
    lower = path.lower()
    if lower.endswith((".csv", ".tsv")):
        return pd.read_csv(path, header=None, sep="\t" if lower.endswith(".tsv") else ",", dtype=str)
    return pd.read_excel(path, header=None)


def preview(datos, rows=8):
    section("DATASET PREVIEW", "データ")
    for index, row in datos.head(rows).iterrows():
        values = c(" │ ", "faint").join(c(str(v), "green") for v in row)
        out(f"  {c(f'R{index:02d}', 'dim')} {c('▸', 'cyan')} {values}")


def choose_header(datos):
    section("HEADER MAPPING", "ヘッダ")
    info("Selecciona la fila con los nombres de columna.")
    while True:
        raw = ask("ENTER HEADER ROW INDEX")
        try:
            idx = int(raw)
            if 0 <= idx < len(datos):
                return idx
            err("INDEX OUT OF RANGE")
        except ValueError:
            err("INVALID NUMBER")


def apply_header(datos, header_row):
    datos.columns = datos.iloc[header_row]
    datos = datos.drop(header_row).reset_index(drop=True)
    datos.columns = [str(col) for col in datos.columns]
    return datos


def is_numeric_series(series):
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    return non_null > 0 and coerced.notna().sum() / non_null >= 0.95


def choose_features(datos):
    section("FEATURE SELECTION", "トクチョウ")
    info("K-means solo usa columnas numericas (distancia euclidiana).")
    numeric = [col for col in datos.columns if is_numeric_series(datos[col])]
    for i, col in enumerate(datos.columns, start=1):
        mark = c("<NUM>", "amber") if col in numeric else c("<skip>", "faint")
        out(f"  {c(f'{i:02d}', 'dim')} {c('▸', 'cyan')} {c(col, 'white')}  {mark}")
    raw = ask("FEATURE COLUMN NUMBERS (coma) [enter = todas las numericas]")
    if not raw:
        return numeric
    chosen = []
    for tok in raw.split(","):
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(datos.columns):
                chosen.append(datos.columns[idx])
        except ValueError:
            pass
    return chosen or numeric


def label_column(datos, features):
    """Columna de nombres/etiqueta para mostrar (primera no-feature)."""
    for col in datos.columns:
        if col not in features:
            return col
    return None


def show_clusters(names, labels, k, centroids, features):
    section("CLUSTER ASSIGNMENT", "ワリテ")
    for cluster in range(k):
        members = [names[i] for i in range(len(names)) if labels[i] == cluster]
        out(f"  {c(f'CLUSTER {cluster + 1}', 'cyan', True)} {c('▸', 'cyan')} "
            f"{c(', '.join(map(str, members)) or '(vacio)', 'green')}")
        ctr = ", ".join(f"{v:.3f}" for v in centroids[cluster])
        out(f"      {c('centro', 'dim')} {c('=', 'faint')} {c('(' + ctr + ')', 'teal')}  "
            f"{c('[' + ' · '.join(features) + ']', 'faint')}")


def main():
    global _COLOR
    enable_ansi()
    _COLOR = _supports_color()
    banner()

    if ask("CLEAR SCREEN? [Y/N]").upper() == "Y":
        clear_console()
        banner()
    boot_sequence()

    section("DATA INGEST", "ロード")
    info("Formatos: .xlsx / .xls / .csv / .tsv")
    path = ask("ENTER DATASET PATH")
    try:
        info("Leyendo fuente de datos...")
        datos = load_dataset(path)
        ok("SOURCE LOADED")
    except FileNotFoundError:
        die("FILE NOT FOUND")
    except Exception as e:
        die(f"LOAD FAILURE — {e}")

    datos = datos.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if datos.empty:
        die("DATASET IS EMPTY AFTER CLEANING")
    ok("EMPTY ROWS / COLUMNS PURGED")

    preview(datos)
    datos = apply_header(datos, choose_header(datos))

    features = choose_features(datos)
    if not features:
        die("NO NUMERIC FEATURES SELECTED")
    ok(f"FEATURES: {', '.join(features)}")

    names_col = label_column(datos, features)
    names = list(datos[names_col]) if names_col else list(range(1, len(datos) + 1))

    section("NORMALIZATION", "セイキカ")
    info("Regla de 3 con el maximo = 1 (metodo del documento).")
    norm = normalize_by_max(datos, features)
    for i, row in norm.head(8).iterrows():
        vals = ", ".join(f"{float(row[f]):.2f}" for f in features)
        out(f"  {c(str(names[i]), 'white'):<20} {c('(' + vals + ')', 'teal')}")
    points = [[float(norm.iloc[i][f]) for f in features] for i in range(len(norm))]

    section("CLUSTER COUNT", "クラスタ")
    raw = ask("NUMBER OF CLUSTERS k [default 2]")
    try:
        k = max(2, int(raw)) if raw else 2
    except ValueError:
        k = 2
    ok(f"k = {k}")

    init = None
    if ask("PROVIDE INITIAL CENTROIDS MANUALLY? [Y/N] (N = aleatorios)").upper() == "Y":
        init = []
        for cluster in range(k):
            vals = ask(f"CENTROID {cluster + 1} — {len(features)} valores separados por coma")
            try:
                init.append([float(x) for x in vals.split(",")])
            except ValueError:
                die("INVALID CENTROID VALUES")

    section("ITERATION", "シュウソク")
    info("Kau'yon: reasignar y actualizar hasta que los centros no cambien.")
    labels, centroids, history = kmeans(points, k, init)
    for step, lbls, ctrs, tau_va in history:
        out(f"  {c(f'iter {step:02d}', 'dim')} {c('│', 'faint')} "
            f"{c(f'tau_va(inercia)={tau_va:.4f}', 'green')}")
    ok(f"CONVERGED IN {len(history)} ITERATIONS")

    show_clusters(names, labels, k, centroids, features)

    print()
    scanline()
    tw("  CLUSTERING COMPLETE // FOR THE GREATER GOOD", 0.004, "cyan", True)
    scanline()


if __name__ == "__main__":
    main()
