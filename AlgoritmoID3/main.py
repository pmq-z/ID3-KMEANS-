import sys
import os
import time
import math

import pandas as pd

# Rama de respaldo para valores nunca vistos en entrenamiento: como un Newtype,
# el arbol percibe lo que no estaba ahi antes y responde con la clase mayoritaria.
NEWTYPE_KEY = "\x00newtype"
MISSING_TOKEN = "?"


# ============================ CAPA LOGICA — ID3 ============================

def entropy(series):
    """Entropia de Shannon (bits)."""
    total = len(series)
    if total == 0:
        return 0.0
    result = 0.0
    for count in series.value_counts():
        p = count / total
        if p > 0:
            result -= p * math.log2(p)
    return result


def max_entropy(series):
    """Entropia maxima posible: log2(k)."""
    k = series.nunique()
    return math.log2(k) if k > 1 else 0.0


def information_gain(dataset, attribute, target):
    """Ganancia de informacion de un atributo respecto al objetivo."""
    total = len(dataset)
    if total == 0:
        return 0.0
    weighted = 0.0
    for value in dataset[attribute].unique():
        subset = dataset[dataset[attribute] == value]
        weighted += (len(subset) / total) * entropy(subset[target])
    return entropy(dataset[target]) - weighted


def is_numeric_series(series):
    """True si casi toda la columna es numerica."""
    coerced = pd.to_numeric(series, errors="coerce")
    non_null = series.notna().sum()
    return non_null > 0 and coerced.notna().sum() / non_null >= 0.95


def discretize_series(series, bins=3, method="width"):
    """Convierte una columna numerica en rangos categoricos.
    method='width' -> ancho igual (pd.cut); 'quantile' -> igual frecuencia (pd.qcut).
    Devuelve (serie_etiquetada, binner) donde binner(x) mapea un numero crudo a su rango."""
    numeric = pd.to_numeric(series, errors="coerce")
    if method == "quantile":
        try:
            cats = pd.qcut(numeric, q=bins, duplicates="drop")
        except (ValueError, IndexError):
            cats = pd.cut(numeric, bins=bins)
    else:
        cats = pd.cut(numeric, bins=bins)

    intervals = cats.cat.categories

    def label(iv):
        return f"[{round(iv.left, 2)}, {round(iv.right, 2)}]"

    labeled = cats.map(lambda iv: MISSING_TOKEN if pd.isna(iv) else label(iv)).astype(str)

    def binner(x):
        v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
        if pd.isna(v):
            return MISSING_TOKEN
        for iv in intervals:
            if v in iv:
                return label(iv)
        return label(intervals[0]) if v <= intervals[0].left else label(intervals[-1])

    return labeled, binner


def id3(dataset, attributes, target, max_depth=None, depth=0):
    """Construye el arbol ID3. Hoja = etiqueta; nodo = {atributo: {valor: subarbol, NEWTYPE_KEY: mayoria}}."""
    labels = dataset[target]
    if labels.nunique() == 1:
        return labels.iloc[0]
    if not attributes or (max_depth is not None and depth >= max_depth):
        return labels.mode()[0]

    gains = {a: information_gain(dataset, a, target) for a in attributes}
    best = max(gains, key=gains.get)
    if gains[best] <= 0:
        return labels.mode()[0]

    majority = labels.mode()[0]
    branches = {NEWTYPE_KEY: majority}
    remaining = [a for a in attributes if a != best]
    for value in sorted(dataset[best].unique(), key=str):
        subset = dataset[dataset[best] == value]
        branches[value] = majority if subset.empty else id3(subset, remaining, target, max_depth, depth + 1)
    return {best: branches}


def classify(tree, instance):
    """Clasifica un ejemplo (dict). Valor no visto -> clase mayoritaria del nodo."""
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    branches = tree[attribute]
    value = instance.get(attribute)
    if value in branches and value != NEWTYPE_KEY:
        return classify(branches[value], instance)
    return branches.get(NEWTYPE_KEY)


def extract_rules(tree, path=None):
    """Deriva la lista de reglas del arbol: [(condiciones, clase), ...]."""
    path = path or []
    if not isinstance(tree, dict):
        return [(path, tree)]
    attribute = next(iter(tree))
    rules = []
    for value, subtree in tree[attribute].items():
        if value == NEWTYPE_KEY:
            continue
        rules += extract_rules(subtree, path + [(attribute, value)])
    return rules


def count_nodes(tree):
    """(nodos_de_decision, hojas)."""
    if not isinstance(tree, dict):
        return 0, 1
    decision, leaves = 1, 0
    for branches in tree.values():
        for key, subtree in branches.items():
            if key == NEWTYPE_KEY:
                continue
            d, l = count_nodes(subtree)
            decision += d
            leaves += l
    return decision, leaves


def tree_depth(tree):
    if not isinstance(tree, dict):
        return 0
    depths = []
    for branches in tree.values():
        for key, subtree in branches.items():
            if key != NEWTYPE_KEY:
                depths.append(tree_depth(subtree))
    return 1 + max(depths) if depths else 1


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
    """Teletipo (solo para lineas de flavor)."""
    if color and _COLOR:
        sys.stdout.write(("\033[1m" if bold else "") + f"\033[{_PAL.get(color, '0')}m")
    for char in str(text):  # Char: el piloto de cada caracter
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
    jp_lbl = c(jp, "teal")
    visible = len(f"╭─╴ {title} {jp}")
    fill = max(0, W - visible - 1)
    out(f"{c('╭─╴', 'cyan')} {label} {jp_lbl} {c('╶' * fill + '╮', 'cyan')}")


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
        "╔══════════════════════════════════════════════════════════╗",
        "║                                                          ║",
        "║      I D 3   D E C I S I O N   T R E E   E N G I N E     ║",
        "║      ─────────────────────────────────────────────       ║",
        "║          決定木 // PROJECT 2501 // PROTOCOL v3.0         ║",
        "║                                                          ║",
        "╚══════════════════════════════════════════════════════════╝",
    ]
    print()
    for i, line in enumerate(art):
        tw(line, 0.0009, "cyan" if i in (2, 4) else "teal", bold=(i == 2))


def boot_sequence():
    steps = [
        "MOUNTING CORE MODULES", "LINKING DATA-PROCESSING BUS",
        "LOADING ID3 INFERENCE KERNEL", "CALIBRATING ENTROPY GAUGE",
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


def preview(datos, rows=6):
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
    seen, dupes = set(), set()
    for col in datos.columns:
        (dupes if col in seen else seen).add(col)
    if dupes:
        die(f"DUPLICATE COLUMN HEADERS: {', '.join(sorted(dupes))}")
    return datos


def show_attributes(datos):
    section("DETECTED ATTRIBUTES", "ゾクセイ")
    for i, col in enumerate(datos.columns, start=1):
        kind = "NUM" if is_numeric_series(datos[col]) else "CAT"
        out(f"  {c(f'{i:02d}', 'dim')} {c('▸', 'cyan')} {c(col, 'white')}  "
            f"{c(f'<{kind}>', 'amber' if kind == 'NUM' else 'teal')}")


def choose_target(datos):
    section("TARGET SELECTION", "モクヒョウ")
    info("Numero de la columna a predecir (clase objetivo).")
    while True:
        raw = ask("ENTER TARGET COLUMN NUMBER")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(datos.columns):
                return datos.columns[idx]
            err("INDEX OUT OF RANGE")
        except ValueError:
            err("INVALID NUMBER")


def handle_numeric_attributes(datos, attributes):
    """Discretiza atributos numericos. Devuelve (datos, binners) para clasificar numeros crudos luego."""
    numeric_attrs = [a for a in attributes if is_numeric_series(datos[a])]
    binners = {}
    if not numeric_attrs:
        return datos, binners

    section("NUMERIC ATTRIBUTE HANDLING", "スウチ")
    warn("ID3 es categorico. Atributos numericos detectados:")
    for a in numeric_attrs:
        out(f"      {c('▸', 'amber')} {c(a, 'white')}")
    if ask("DISCRETIZE NUMERIC ATTRIBUTES? [Y/N]").upper() != "Y":
        warn("Se mantienen como categoricos (posible sobreajuste).")
        return datos, binners

    method = "quantile" if ask("METHOD: [W]idth (ancho igual) / [Q]uantile").upper().startswith("Q") else "width"
    bins = 3
    raw = ask("NUMBER OF BINS PER ATTRIBUTE [default 3]")
    if raw:
        try:
            bins = max(2, int(raw))
        except ValueError:
            warn("Valor invalido; usando 3.")

    label_method = "ancho igual" if method == "width" else "cuantiles"
    for a in numeric_attrs:
        datos[a], binners[a] = discretize_series(datos[a], bins, method)
        ok(f"'{a}' -> {bins} rangos ({label_method}).")
    return datos, binners


def report_gains(datos, attributes, target):
    total = len(datos)
    H = entropy(datos[target])
    Hmax = max_entropy(datos[target])
    section("ENTROPY DIAGNOSTICS", "エントロピー")
    field("DATASET ENTROPY", f"{H:.4f} bits")
    field("MAX ENTROPY log2(k)", f"{Hmax:.4f} bits")
    ratio = H / Hmax if Hmax > 0 else 0.0
    field("NORMALIZED (H / Hmax)", f"{ratio:.2%}")
    if H == 0.0:
        die("ENTROPY IS ZERO — DATASET ALREADY PURE")
    info("Clasificacion muy mezclada." if ratio >= 0.85 else "Clasificacion moderadamente ordenada.")

    section("INFORMATION GAIN PER ATTRIBUTE", "リエキ")
    gains = {}
    for a in attributes:
        gains[a] = information_gain(datos, a, target)
        out(f"  {c('▸', 'cyan')} {c(a, 'white', True)}")
        for value in sorted(datos[a].unique(), key=str):
            subset = datos[datos[a] == value]
            out(f"      {c(str(value), 'teal'):<22} {c(f'n={len(subset):>3}', 'dim')}  "
                f"{c(f'H={entropy(subset[target]):.4f}', 'dim')}  {c(f'w={len(subset)/total:.3f}', 'dim')}")
        out(f"      {c('GAIN', 'amber')} {c('=', 'faint')} {c(f'{gains[a]:.4f}', 'green', True)}")
        print()
    return gains


def rank_attributes(gains):
    section("ROOT ATTRIBUTE RANKING", "ルート")
    best = max(gains, key=gains.get)
    top = max(gains.values()) or 1
    for a, g in sorted(gains.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(round(g / top * 24))
        bar = c("█" * bar_len, "cyan") + c("░" * (24 - bar_len), "faint")
        marker = c("  ◄ ROOT", "amber", True) if a == best else ""
        out(f"  {c(a, 'white'):<22} {bar} {c(f'{g:.4f}', 'green')}{marker}")


def render_tree(tree, prefix="", is_root=True):
    lines = []
    if not isinstance(tree, dict):
        return [prefix + c("▸ ", "faint") + c(str(tree), "green", True)]
    attribute = next(iter(tree))
    branches = [(k, v) for k, v in tree[attribute].items() if k != NEWTYPE_KEY]
    if is_root:
        lines.append(c("┌─┤ ", "cyan") + c(str(attribute), "cyan", True) + c(" ├─", "cyan"))
    for i, (value, subtree) in enumerate(branches):
        last = i == len(branches) - 1
        conn = "└── " if last else "├── "
        child_prefix = prefix + ("    " if last else "│   ")
        line = prefix + c(conn, "faint") + c(str(value), "teal")
        if isinstance(subtree, dict):
            sub_attr = next(iter(subtree))
            line += c("  ┤ ", "cyan") + c(str(sub_attr), "cyan", True) + c(" ├", "cyan")
            lines.append(line)
            lines.extend(render_tree(subtree, child_prefix, False))
        else:
            lines.append(line + c("  ▸ ", "faint") + c(str(subtree), "green", True))
    return lines


def show_rules(tree, target):
    section("DECISION RULES", "ルール")
    for conditions, result in extract_rules(tree):
        antecedent = c(" ∧ ", "faint").join(
            f"{c(a, 'white')}{c('=', 'faint')}{c(v, 'teal')}" for a, v in conditions)
        out(f"  {c('SI', 'cyan', True)} {antecedent} {c('⇒', 'cyan')} "
            f"{c(target, 'white')}{c('=', 'faint')}{c(str(result), 'magenta', True)}")


def interactive_classify(tree, attributes, datos, binners):
    section("INFERENCE MODE", "スイソク")
    info("Clasifica ejemplos nuevos. Enter vacio para salir.")
    domains = {a: sorted(datos[a].unique(), key=str) for a in attributes}
    while True:
        print()
        out(c("  ── NEW SAMPLE ──", "cyan", True))
        sortie = {}  # cada ejemplo es una salida al frente
        aborted = False
        for a in attributes:
            if a in binners:
                out(f"  {c(a, 'white')} {c('(numero crudo)', 'faint')}")
                val = ask(f"VALUE FOR '{a}'  [enter=quit]")
                if val == "":
                    aborted = True
                    break
                sortie[a] = binners[a](val)  # numero -> rango automatico
            else:
                out(f"  {c(a, 'white')} {c('(' + ', '.join(map(str, domains[a])) + ')', 'faint')}")
                val = ask(f"VALUE FOR '{a}'  [enter=quit]")
                if val == "":
                    aborted = True
                    break
                sortie[a] = val
        if aborted:
            break
        result = classify(tree, sortie)
        print()
        out(f"  {c('╢', 'cyan')} PREDICTION {c('▸', 'cyan')} "
            f"{c(str(result), 'magenta', True)} {c('╟', 'cyan')}")
    ok("INFERENCE SESSION CLOSED")


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

    section("DATASET INFORMATION", "ジョウホウ")
    field("ROWS", str(datos.shape[0]))
    field("COLUMNS", str(datos.shape[1]))
    show_attributes(datos)

    if ask("CONTINUE WITH THIS DATASET? [Y/N]").upper() != "Y":
        die("OPERATION ABORTED BY OPERATOR")
    ok("DATASET VALIDATED")

    target = choose_target(datos)
    ok(f"TARGET LOCKED: {target}")
    attributes = [col for col in datos.columns if col != target]

    before = len(datos)
    datos = datos.dropna(subset=[target]).reset_index(drop=True)
    if len(datos) < before:
        warn(f"{before - len(datos)} filas sin clase descartadas.")
    for a in attributes:
        datos[a] = datos[a].fillna(MISSING_TOKEN).astype(str)
    datos[target] = datos[target].astype(str)
    if datos.empty:
        die("NO LABELED ROWS AVAILABLE")

    datos, binners = handle_numeric_attributes(datos, attributes)

    section("CLASS DISTRIBUTION", "ブンプ")
    total = len(datos)
    for label, count in datos[target].value_counts().items():
        pct = count / total * 100
        bar_len = int(round(pct / 100 * 20))
        bar = c("█" * bar_len, "green") + c("░" * (20 - bar_len), "faint")
        out(f"  {c(str(label), 'white'):<16} {bar} {c(f'{count:>4}', 'teal')} {c(f'({pct:4.1f}%)', 'dim')}")

    gains = report_gains(datos, attributes, target)
    rank_attributes(gains)

    section("TREE SYNTHESIS", "コウチク")
    info("Ejecutando ID3...")
    time.sleep(0.2)
    tree = id3(datos, attributes, target)
    ok("DECISION TREE GENERATED")
    decision_nodes, leaves = count_nodes(tree)
    field("DEPTH", str(tree_depth(tree)))
    field("DECISION NODES", str(decision_nodes))
    field("LEAF NODES", str(leaves))
    field("TOTAL NODES", str(decision_nodes + leaves))

    section("DECISION TREE", "ケッカ")
    print()
    for line in render_tree(tree):
        out(line)

    show_rules(tree, target)
    print()
    scanline()

    if ask("ENTER INFERENCE MODE TO CLASSIFY NEW SAMPLES? [Y/N]").upper() == "Y":
        interactive_classify(tree, attributes, datos, binners)

    print()
    scanline()
    tw("  PROCESS COMPLETE // GHOST DISCONNECTED", 0.004, "cyan", True)
    scanline()


if __name__ == "__main__":
    main()
