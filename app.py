"""Gestor de textos con búsqueda por similitud de coseno y explicación.

Tabs:
  - Bulk data: insertar, editar, borrar, eliminar todo y cargar textos desde un .txt
  - Búsqueda por cosenos: escribe un texto y ve la similitud contra cada texto guardado
  - Explicación: por qué un texto quedó seleccionado (qué elementos coinciden)

Uso:
    pip install flask ollama
    ollama pull nomic-embed-text      (y tener el servidor de Ollama corriendo)
    python app.py
    Abrir http://127.0.0.1:5000
"""
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import uuid

import numpy as np

from flask import Flask, flash, redirect, render_template_string, request, session, url_for

try:
    import mlflow
except Exception:  # MLflow es opcional; si no está instalado, la app sigue funcionando.
    mlflow = None

# ---------- Configuración ----------
MODELO = "nomic-embed-text"
PREFIJO = "clustering: "  # el mismo prefijo para consulta y textos
LOTE = 32                 # textos por llamada a Ollama
MAX_PALABRAS = 40         # palabras que se analizan por texto en la pestaña Explicación
NOMBRE_EXPERIMENTO_MLFLOW = "similitud"

DEFAULT_TUNING = {
    "modelo": MODELO,
    "prefijo": PREFIJO,
    "lote": LOTE,
    "stopwords": "el la los las un una unos unas de del al y o u e en a con por para que se su sus lo le les es son fue era ser mi tu mis tus nos me te si no ni como mas pero ya este esta estos estas ese esa eso muy sin sobre entre hay",
    "max_palabras": MAX_PALABRAS,
    "metrica": "coseno",
    "preprocesado": "normal",
}

app = Flask(__name__)
app.secret_key = "cambia-esto-si-lo-publicas"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB máximo por archivo


def preparar_experimento_mlflow():
    """Asegura que exista un experimento activo para registrar runs."""
    if mlflow is None:
        return None

    try:
        mlflow.set_tracking_uri("http://localhost:5003")
    except Exception:
        pass

    nombre = NOMBRE_EXPERIMENTO_MLFLOW
    try:
        exp = mlflow.get_experiment_by_name(nombre)
        if exp is not None:
            if exp.lifecycle_stage == "active":
                mlflow.set_experiment(nombre)
                return exp.experiment_id
            try:
                mlflow.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(nombre)
                exp = mlflow.get_experiment_by_name(nombre)
                if exp is not None and exp.lifecycle_stage == "active":
                    return exp.experiment_id
            except Exception:
                pass

        try:
            nuevo_id = mlflow.create_experiment(nombre)
            mlflow.set_experiment(nombre)
            return nuevo_id
        except Exception:
            try:
                mlflow.set_experiment(nombre)
                exp2 = mlflow.get_experiment_by_name(nombre)
                if exp2 is not None:
                    return exp2.experiment_id
            except Exception:
                pass
            return None
    except Exception:
        return None


if mlflow is not None:
    # El flujo de la app se sirve en 5001; MLflow debe vivir en otro puerto para no chocar.
    preparar_experimento_mlflow()

BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(BASE, "textos.json")
CACHE = os.path.join(BASE, "embeddings.json")

# Palabras que no cuentan como "elementos" al buscar coincidencias
VACIAS = set(
    "el la los las un una unos unas de del al y o u e en a con por para que se su sus lo le "
    "les es son fue era ser mi tu mis tus nos me te si no ni como mas pero ya este esta estos "
    "estas ese esa eso muy sin sobre entre hay".split()
)


# ---------- Persistencia ----------
def leer_json(ruta, defecto):
    if not os.path.exists(ruta):
        return defecto
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def escribir_json(ruta, datos):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False)


def cargar():
    return leer_json(ARCHIVO, [])


def guardar(items):
    with open(ARCHIVO, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def nuevo_item(texto):
    return {"id": uuid.uuid4().hex, "texto": texto}


def listar_modelos_ollama():
    """Devuelve solo los modelos compatibles con embeddings en Ollama."""
    try:
        import ollama
        datos = ollama.list()
        modelos = []
        if hasattr(datos, "models"):
            modelos = list(getattr(datos, "models", []))
        elif isinstance(datos, dict):
            modelos = datos.get("models", []) or []
        elif isinstance(datos, list):
            modelos = datos

        compatibles = []
        for item in modelos:
            if isinstance(item, dict):
                nombre = item.get("name") or item.get("model")
            elif hasattr(item, "model"):
                nombre = item.model
            elif isinstance(item, str):
                nombre = item
            else:
                continue
            if not nombre:
                continue

            texto = nombre.lower()
            try:
                info = ollama.show(nombre)
            except Exception:
                if any(token in texto for token in ("embed", "embedding", "all-minilm", "bge", "nomic")):
                    compatibles.append(nombre)
                continue

            caps = []
            if hasattr(info, "capabilities"):
                caps = list(getattr(info, "capabilities", []) or [])
            elif isinstance(info, dict):
                caps = info.get("capabilities") or []
            if isinstance(caps, str):
                caps = [caps]
            caps_text = " ".join(str(c).lower() for c in caps)

            family = ""
            details = getattr(info, "details", None)
            if details is not None:
                family = getattr(details, "family", "") or ""
                if not isinstance(family, str):
                    family = str(family)
            elif isinstance(info, dict):
                details = info.get("details") or {}
                family = str(details.get("family", "") or "")

            es_embedding = (
                "embed" in caps_text or "embedding" in caps_text or
                "embed" in texto or "embedding" in texto or
                "all-minilm" in texto or "bge" in texto or "nomic" in texto or
                "bert" in family.lower() or "nomic-bert" in family.lower() or "all-minilm" in family.lower()
            )

            if es_embedding:
                compatibles.append(nombre)

        if compatibles:
            return list(dict.fromkeys(compatibles))
    except Exception:
        pass
    return [MODELO]


def modelo_actual(modelos=None):
    """Modelo activo para la sesión actual."""
    cfg = config_actual(modelos)
    return cfg["modelo"]


def config_actual(modelos=None):
    """Lee la configuración actual desde la sesión y los argumentos de la petición."""
    disponibles = modelos or listar_modelos_ollama()
    base = dict(DEFAULT_TUNING)
    guardada = session.get("tuning", {}) or {}
    base.update({k: v for k, v in guardada.items() if v is not None})

    for clave, valor in {
        "modelo": request.args.get("modelo", "").strip(),
        "prefijo": request.args.get("prefijo", "").strip(),
        "lote": request.args.get("lote", "").strip(),
        "stopwords": request.args.get("stopwords", "").strip(),
        "max_palabras": request.args.get("max_palabras", "").strip(),
        "metrica": request.args.get("metrica", "").strip(),
        "preprocesado": request.args.get("preprocesado", "").strip(),
    }.items():
        if valor not in (None, ""):
            base[clave] = valor

    try:
        base["lote"] = int(base.get("lote") or LOTE)
    except Exception:
        base["lote"] = LOTE
    base["lote"] = max(1, min(base["lote"], 256))

    try:
        base["max_palabras"] = int(base.get("max_palabras") or MAX_PALABRAS)
    except Exception:
        base["max_palabras"] = MAX_PALABRAS
    base["max_palabras"] = max(1, min(base["max_palabras"], 200))

    if base.get("modelo") not in disponibles:
        base["modelo"] = disponibles[0] if disponibles else MODELO

    if not base.get("prefijo"):
        base["prefijo"] = PREFIJO
    if base.get("metrica") not in {"coseno", "euclidiana", "manhattan"}:
        base["metrica"] = "coseno"
    if base.get("preprocesado") not in {"normal", "sin_acentos", "sin_puntuacion", "sin_acentos_ni_puntuacion"}:
        base["preprocesado"] = "normal"

    session["modelo_seleccionado"] = base["modelo"]
    session["tuning"] = {
        "modelo": base["modelo"],
        "prefijo": base["prefijo"],
        "lote": base["lote"],
        "stopwords": base["stopwords"],
        "max_palabras": base["max_palabras"],
        "metrica": base["metrica"],
        "preprocesado": base["preprocesado"],
    }
    return base


def stopwords_config(config=None):
    cfg = config or config_actual()
    valores = str(cfg.get("stopwords") or "").lower()
    tokens = re.split(r"[\s,;]+", valores)
    return {t for t in tokens if t}


# ---------- Embeddings y coseno ----------
def firma(texto, modelo=None, prefijo=None):
    """Cambia si cambia el texto, el modelo o el prefijo: así se recalcula solo."""
    modelo = modelo or MODELO
    prefijo = prefijo or PREFIJO
    return hashlib.sha1(f"{modelo}|{prefijo}|{texto}".encode("utf-8")).hexdigest()


def embeber(textos, modelo=None, prefijo=None):
    import ollama  # import aquí para que la pestaña Bulk data funcione sin Ollama

    modelo = modelo or MODELO
    prefijo = prefijo or PREFIJO
    respuesta = ollama.embed(model=modelo, input=[prefijo + t for t in textos])
    return respuesta["embeddings"]


def vectores_de(items, modelo=None, prefijo=None, lote=None):
    """Devuelve {id: vector}. Solo calcula los que faltan o cambiaron."""
    modelo = modelo or MODELO
    prefijo = prefijo or PREFIJO
    lote = lote or LOTE
    cache = leer_json(CACHE, {})
    pendientes = [it for it in items if cache.get(it["id"], {}).get("firma") != firma(it["texto"], modelo=modelo, prefijo=prefijo)]

    for i in range(0, len(pendientes), lote):
        grupo = pendientes[i:i + lote]
        for it, vec in zip(grupo, embeber([g["texto"] for g in grupo], modelo=modelo, prefijo=prefijo)):
            cache[it["id"]] = {"firma": firma(it["texto"], modelo=modelo, prefijo=prefijo), "vec": vec, "modelo": modelo}

    vigentes = {it["id"] for it in items}
    antes = len(cache)
    cache = {k: v for k, v in cache.items() if k in vigentes}  # limpia borrados
    if pendientes or len(cache) != antes:
        escribir_json(CACHE, cache)
    return {k: v["vec"] for k, v in cache.items() if v.get("modelo") == modelo}


def coseno(a, b):
    punto = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return punto / (na * nb) if na and nb else 0.0


def similitud(a, b, metrica="coseno"):
    """Calcula la similitud según la métrica activa.

    Coseno devuelve un valor alto para textos parecidos; Euclidiana y Manhattan
    usan distancias, así que una distancia menor significa mayor similitud.
    """
    if metrica == "euclidiana":
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        # Invertimos la distancia para que valores más altos signifiquen mayor semejanza.
        return -float(np.linalg.norm(diff))
    if metrica == "manhattan":
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        # La distancia Manhattan también se invierte para mantener la misma lógica visual.
        return -float(np.abs(diff).sum())
    return coseno(a, b)


def puntuar(consulta, items, modelo=None, prefijo=None, metrica="coseno"):
    """Similitud de la consulta contra todos los textos, de mayor a menor."""
    modelo = modelo or MODELO
    prefijo = prefijo or PREFIJO
    metrica = metrica or "coseno"
    lote_cfg = int((session.get("tuning", {}) or {}).get("lote", LOTE) or LOTE)
    vectores = vectores_de(items, modelo=modelo, prefijo=prefijo, lote=max(1, lote_cfg))
    vec_q = embeber([consulta], modelo=modelo, prefijo=prefijo)[0]
    ranking = [
        {"id": it["id"], "texto": it["texto"], "score": similitud(vec_q, vectores[it["id"]], metrica=metrica)}
        for it in items
    ]
    ranking.sort(key=lambda r: r["score"], reverse=True)
    return ranking, vec_q


# ---------- Explicación ----------
def preprocesar_texto(texto, modo="normal"):
    valor = str(texto)
    if modo in {"sin_acentos", "sin_acentos_ni_puntuacion"}:
        valor = unicodedata.normalize("NFD", valor.lower())
        valor = "".join(c for c in valor if unicodedata.category(c) != "Mn")
    else:
        valor = valor.lower()

    if modo in {"sin_puntuacion", "sin_acentos_ni_puntuacion"}:
        valor = re.sub(r"[^\w\s]", " ", valor)
    return valor


def normalizar(palabra):
    sin_acentos = unicodedata.normalize("NFD", palabra.lower())
    return "".join(c for c in sin_acentos if unicodedata.category(c) != "Mn")


def misma_raiz(a, b):
    """Iguales, o variantes de la misma palabra (corre/corren, niño/niños)."""
    if a == b:
        return True
    corta, larga = sorted((a, b), key=len)
    return len(corta) >= 4 and larga.startswith(corta) and len(larga) - len(corta) <= 3


def elementos_en_comun(texto, consulta, stopwords=None, preprocesado="normal"):
    """Palabras del texto que coinciden con palabras de la consulta."""
    stopwords = stopwords if stopwords is not None else VACIAS

    def contenido(s):
        procesado = preprocesar_texto(s, preprocesado)
        return [(p, normalizar(p)) for p in re.findall(r"\w+", procesado)
                if len(p) >= 3 and normalizar(p) not in stopwords]

    palabras_q = contenido(consulta)
    vistos, comunes = set(), []
    for original, norm in contenido(texto):
        if norm in vistos:
            continue
        for orig_q, norm_q in palabras_q:
            if misma_raiz(norm, norm_q):
                vistos.add(norm)
                comunes.append({"texto": original, "consulta": orig_q})
                break
    return comunes


def aporte_por_palabra(texto, vec_q, score_base, modelo=None, prefijo=None, lote=None, metrica="coseno", max_palabras=None):
    """Cuánto baja la similitud al quitar cada palabra (explicación por oclusión)."""
    modelo = modelo or MODELO
    prefijo = prefijo or PREFIJO
    lote = lote or LOTE
    metrica = metrica or "coseno"
    max_palabras = max_palabras or MAX_PALABRAS
    spans = [(m.start(), m.end()) for m in re.finditer(r"\w+", texto)][:max(1, int(max_palabras))]
    if len(spans) < 2:
        return spans, []
    variantes = [re.sub(r"\s+", " ", texto[:s] + texto[e:]).strip() for s, e in spans]
    vecs = []
    for i in range(0, len(variantes), lote):
        vecs.extend(embeber(variantes[i:i + lote], modelo=modelo, prefijo=prefijo))
    return spans, [score_base - similitud(vec_q, v, metrica=metrica) for v in vecs]


def segmentar(texto, spans, deltas):
    """Parte el texto en trozos para resaltar cada palabra según su aporte."""
    segmentos, pos = [], 0
    maximo = max((abs(d) for d in deltas), default=0) or 1
    for (s, e), d in zip(spans, deltas):
        if s > pos:
            segmentos.append({"t": texto[pos:s], "estilo": "", "tip": ""})
        alfa = min(0.6, abs(d) / maximo * 0.6)
        color = "15,107,92" if d > 0 else "179,55,43"
        estilo = f"background: rgba({color},{alfa:.2f})" if alfa >= 0.06 else ""
        segmentos.append({"t": texto[s:e], "estilo": estilo, "tip": f"Aporte {d:+.4f}"})
        pos = e
    if pos < len(texto):
        segmentos.append({"t": texto[pos:], "estilo": "", "tip": ""})
    return segmentos


def plot_embedding(items, consulta, modelo, id_sel=None, prefijo=None):
    """Genera una proyección 2D en SVG usando PCA para visualizar el espacio semántico."""
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return None

    if not items:
        return None

    prefijo = prefijo or PREFIJO
    vectores = vectores_de(items, modelo=modelo, prefijo=prefijo)
    if not vectores:
        return None

    consulta_vec = embeber([consulta], modelo=modelo, prefijo=prefijo)[0]
    matriz = []
    nombres = []
    ids = []
    seleccion = []

    for it in items:
        vid = it["id"]
        vec = vectores.get(vid)
        if vec is None:
            continue
        matriz.append(vec)
        nombres.append(it["texto"][:34])
        ids.append(vid)
        seleccion.append(vid == (id_sel or ""))

    if len(matriz) < 2:
        return None

    matriz = np.asarray(matriz, dtype=float)
    if matriz.shape[0] < 2:
        return None

    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.vstack([consulta_vec, matriz]))
    coords_q = coords[0]
    coords_doc = coords[1:]

    xs = coords_doc[:, 0]
    ys = coords_doc[:, 1]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    def normalizar(valor, lo, hi, margen=12):
        if hi == lo:
            return 50
        escala = 76 / (hi - lo)
        return margen + (valor - lo) * escala

    puntos = []
    for idx, (x, y) in enumerate(coords_doc):
        puntos.append({
            "id": ids[idx],
            "texto": nombres[idx],
            "x": normalizar(x, min_x, max_x),
            "y": normalizar(y, min_y, max_y),
            "selected": seleccion[idx],
        })

    qx = normalizar(coords_q[0], min_x, max_x)
    qy = normalizar(coords_q[1], min_y, max_y)

    return {
        "query": {"x": qx, "y": qy},
        "puntos": puntos,
        "modelo": modelo,
        "ejes": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
    }


def armar_explicacion(consulta, ranking, vec_q, elegido, modelo=None, prefijo=None, lote=None, max_palabras=None, stopwords=None, metrica="coseno", preprocesado="normal"):
    cfg = config_actual()
    prefijo = prefijo or cfg.get("prefijo", PREFIJO)
    lote = lote or cfg.get("lote", LOTE)
    max_palabras = max_palabras or cfg.get("max_palabras", MAX_PALABRAS)
    stopwords = stopwords if stopwords is not None else stopwords_config(cfg)
    metrica = metrica or cfg.get("metrica", "coseno")
    preprocesado = preprocesado or cfg.get("preprocesado", "normal")

    posicion = ranking.index(elegido) + 1
    otros = [r["score"] for r in ranking if r is not elegido]
    media_otros = sum(otros) / len(otros) if otros else None
    comunes = elementos_en_comun(elegido["texto"], consulta, stopwords=stopwords, preprocesado=preprocesado)
    spans, deltas = aporte_por_palabra(
        elegido["texto"], vec_q, elegido["score"],
        modelo=modelo, prefijo=prefijo, lote=lote, metrica=metrica, max_palabras=max_palabras
    )

    palabras = [elegido["texto"][s:e] for s, e in spans]
    top = sorted(zip(palabras, deltas), key=lambda x: x[1], reverse=True)
    top_pos = [(p, d) for p, d in top if d > 0][:5]

    resumen = [f"Obtuvo una similitud de {elegido['score']:.4f}, el lugar {posicion} de {len(ranking)}."]
    if media_otros is not None:
        dif = elegido["score"] - media_otros
        sentido = "por encima" if dif >= 0 else "por debajo"
        resumen.append(f"Los demás textos promedian {media_otros:.4f}, así que este queda {abs(dif):.4f} {sentido}.")
    if comunes:
        lista = ", ".join(c["texto"] for c in comunes)
        resumen.append(f"Coinciden {len(comunes)} elementos con tu consulta: {lista}.")
    else:
        resumen.append("No comparte ninguna palabra con tu consulta, así que la similitud viene del significado general.")
    if top_pos:
        resumen.append("Las palabras que más sostienen la similitud son: "
                       + ", ".join(p for p, _ in top_pos[:3]) + ".")

    return {
        "texto": elegido["texto"],
        "resumen": " ".join(resumen),
        "comunes": comunes,
        "segmentos": segmentar(elegido["texto"], spans, deltas) if deltas else [],
        "top": top_pos,
        "recortado": len(re.findall(r"\w+", elegido["texto"])) > max_palabras,
        "una_palabra": len(spans) < 2,
    }


@app.get("/tuning")
def tuning():
    modelos = listar_modelos_ollama()
    cfg = config_actual(modelos)
    return render_template_string(
        PLANTILLA,
        vista="tuning",
        items=cargar(),
        modelo=cfg["modelo"],
        config=cfg,
    )


@app.post("/tuning")
def tuning_guardar():
    modelos = listar_modelos_ollama()
    modelo = session.get("modelo_seleccionado") or (session.get("tuning", {}) or {}).get("modelo") or MODELO
    if modelo not in modelos:
        modelo = modelos[0] if modelos else MODELO
    cfg = {
        "modelo": modelo,
        "prefijo": request.form.get("prefijo", PREFIJO).strip() or PREFIJO,
        "lote": request.form.get("lote", LOTE).strip() or LOTE,
        "stopwords": request.form.get("stopwords", DEFAULT_TUNING["stopwords"]),
        "max_palabras": request.form.get("max_palabras", MAX_PALABRAS).strip() or MAX_PALABRAS,
        "metrica": request.form.get("metrica", "coseno").strip() or "coseno",
        "preprocesado": request.form.get("preprocesado", "normal").strip() or "normal",
    }
    if cfg["metrica"] not in {"coseno", "euclidiana", "manhattan"}:
        cfg["metrica"] = "coseno"
    if cfg["preprocesado"] not in {"normal", "sin_acentos", "sin_puntuacion", "sin_acentos_ni_puntuacion"}:
        cfg["preprocesado"] = "normal"
    session["tuning"] = cfg
    session["modelo_seleccionado"] = cfg["modelo"]
    flash("Ajustes de tuning guardados.", "ok")
    return redirect(url_for("busqueda", q=session.get("ultima_consulta", ""), modelo=cfg["modelo"]))


# ---------- Rutas: Bulk data ----------
@app.get("/")
def inicio():
    return render_template_string(PLANTILLA, vista="bulk", items=cargar())


@app.get("/descripcion")
def descripcion():
    return render_template_string(
        PLANTILLA,
        vista="descripcion",
        items=cargar(),
        descripcion_proyecto={
            "titulo": "Proyecto de búsqueda semántica con embeddings",
            "texto": "Esta aplicación guarda textos, los convierte en embeddings con Ollama y compara consultas con esos documentos usando similitud coseno. La idea es encontrar textos con significado parecido aunque no compartan exactamente las mismas palabras.",
            "puntos": [
                "Guarda textos en un archivo JSON y los muestra en la pestaña de datos.",
                "Genera embeddings con un modelo compatible de Ollama, como nomic-embed-text o bge-large.",
                "Compara la consulta contra todos los textos usando similitud coseno.",
                "Explica por qué un texto quedó mejor rankeado con palabras clave que coinciden y una proyección visual del espacio vectorial.",
                "Permite comparar varios textos seleccionados para entender mejor la diferencia de score.",
            ],
        },
    )


@app.post("/agregar")
def agregar():
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("Escribe un texto antes de insertar.", "error")
        return redirect(url_for("inicio"))
    items = cargar()
    items.append(nuevo_item(texto))
    guardar(items)
    flash("Texto insertado.", "ok")
    return redirect(url_for("inicio"))


@app.post("/subir")
def subir():
    archivo = request.files.get("archivo")
    if not archivo or not archivo.filename:
        flash("Elige un archivo .txt para subir.", "error")
        return redirect(url_for("inicio"))
    if not archivo.filename.lower().endswith(".txt"):
        flash("Solo se aceptan archivos .txt.", "error")
        return redirect(url_for("inicio"))

    datos = archivo.read()
    try:
        contenido = datos.decode("utf-8-sig")
    except UnicodeDecodeError:
        contenido = datos.decode("latin-1")

    if request.form.get("modo") == "parrafos":
        partes = re.split(r"\n\s*\n", contenido)  # separados por línea en blanco
    else:
        partes = contenido.splitlines()  # un texto por línea

    nuevos = [p.strip() for p in partes if p.strip()]
    if not nuevos:
        flash("El archivo no tiene texto.", "error")
        return redirect(url_for("inicio"))

    items = cargar()
    items.extend(nuevo_item(t) for t in nuevos)
    guardar(items)
    flash(f"Se insertaron {len(nuevos)} textos desde {archivo.filename}.", "ok")
    return redirect(url_for("inicio"))


@app.post("/editar/<item_id>")
def editar(item_id):
    texto = request.form.get("texto", "").strip()
    if not texto:
        flash("El texto no puede quedar vacío. Para quitarlo, usa Borrar.", "error")
        return redirect(url_for("inicio"))
    items = cargar()
    for it in items:
        if it["id"] == item_id:
            it["texto"] = texto
            guardar(items)
            flash("Cambios guardados.", "ok")
            break
    else:
        flash("No se encontró ese texto.", "error")
    return redirect(url_for("inicio"))


@app.post("/borrar/<item_id>")
def borrar(item_id):
    items = cargar()
    restantes = [it for it in items if it["id"] != item_id]
    if len(restantes) == len(items):
        flash("No se encontró ese texto.", "error")
    else:
        guardar(restantes)
        flash("Texto borrado.", "ok")
    return redirect(url_for("inicio"))


@app.post("/borrar-todo")
def borrar_todo():
    cantidad = len(cargar())
    guardar([])
    escribir_json(CACHE, {})
    flash(f"Se borraron {cantidad} textos." if cantidad else "No había textos que borrar.", "ok")
    return redirect(url_for("ianicio"))


def consulta_actual():
    if "q" in request.args:
        consulta = request.args.get("q", "").strip()
        if consulta:
            session["ultima_consulta"] = consulta
        elif "ultima_consulta" in session:
            session.pop("ultima_consulta", None)
        return consulta
    return (session.get("ultima_consulta") or "").strip()


def id_actual():
    if "id" in request.args:
        id_sel = request.args.get("id", "").strip()
        if id_sel:
            session["ultimo_id"] = id_sel
        elif "ultimo_id" in session:
            session.pop("ultimo_id", None)
        return id_sel
    return (session.get("ultimo_id") or "").strip()


# ---------- Ruta: Búsqueda por cosenos ----------
@app.get("/busqueda")
def busqueda():
    modelos = listar_modelos_ollama()
    cfg = config_actual(modelos)
    modelo = cfg["modelo"]
    consulta = consulta_actual()
    items = cargar()
    resultados, error = [], None

    if consulta and items:
        try:
            inicio_calculo = time.perf_counter()
            resultados, _ = puntuar(consulta, items, modelo=modelo, prefijo=cfg["prefijo"], metrica=cfg["metrica"])
            tiempo_resolucion = time.perf_counter() - inicio_calculo

            if mlflow is not None and resultados:
                exp_id = preparar_experimento_mlflow()
                nombre_modelo_run = re.sub(r"[^A-Za-z0-9._-]+", "-", modelo).strip("-") or "modelo"
                with mlflow.start_run(
                    run_name=f"{nombre_modelo_run}-{uuid.uuid4().hex[:8]}",
                    experiment_id=exp_id,
                ):
                    mlflow.log_param("query", consulta)
                    mlflow.log_param("modelo", modelo)
                    mlflow.log_param("metrica", cfg["metrica"])
                    mlflow.log_metric("total_resultados", float(len(resultados)))
                    mlflow.log_metric("max_similitud", float(resultados[0]["score"]))
                    mlflow.log_metric("time_to_complete_seconds", tiempo_resolucion)

                    metricas_registradas = set()
                    for idx, item in enumerate(resultados):
                        rank = idx + 1
                        texto_metrica = unicodedata.normalize(
                            "NFKD", str(item["texto"])
                        ).encode("ascii", "ignore").decode("ascii").lower()
                        texto_metrica = re.sub(r"[^a-z0-9]+", "_", texto_metrica).strip("_")
                        texto_metrica = texto_metrica[:180].strip("_") or f"resultado_{rank}"
                        nombre_metrica = f"similitud_{texto_metrica}"
                        if nombre_metrica in metricas_registradas:
                            nombre_metrica = f"{nombre_metrica}_rank_{rank}"
                        metricas_registradas.add(nombre_metrica)
                        mlflow.log_metric(nombre_metrica, float(item["score"]))
                        mlflow.log_param(
                            f"texto_rank_{rank}_vista_previa",
                            f"{str(item['texto'])[:220]} | similitud={float(item['score']):.6f}",
                        )

                    textos_y_evaluaciones = [
                        {
                            "rank": idx + 1,
                            "texto": item["texto"],
                            "score": float(item["score"]),
                        }
                        for idx, item in enumerate(resultados)
                    ]
                    mlflow.log_dict(
                        {
                            "query": consulta,
                            "modelo": modelo,
                            "metrica": cfg["metrica"],
                            "textos_y_evaluaciones": textos_y_evaluaciones,
                        },
                        "textos_y_evaluaciones.json",
                    )
        except Exception as e:  # Ollama apagado, modelo sin descargar, etc.
            error = (
                f"No se pudo calcular la similitud: {e}. "
                f"Verifica que Ollama esté corriendo y que exista el modelo "
                f"(ollama pull {modelo})."
            )

    return render_template_string(
        PLANTILLA, vista="busqueda", items=items, consulta=consulta,
        resultados=resultados, error=error, modelo=modelo, modelos=modelos,
        metrica=cfg["metrica"],
    )


# ---------- Ruta: Explicación ----------
@app.get("/explicacion")
def explicacion():
    modelos = listar_modelos_ollama()
    cfg = config_actual(modelos)
    modelo = cfg["modelo"]
    consulta = consulta_actual()
    id_sel = id_actual()
    items = cargar()
    selected_ids = request.args.getlist("ids")
    exp, error = None, None

    if selected_ids:
        items = [it for it in items if it["id"] in selected_ids]
        if not items:
            items = cargar()

    if consulta and items:
      try:
        ranking, vec_q = puntuar(consulta, items, modelo=modelo, prefijo=cfg["prefijo"], metrica=cfg["metrica"])
        elegido = next((r for r in ranking if r["id"] == id_sel), ranking[0])
        exp = armar_explicacion(
          consulta, ranking, vec_q, elegido,
          modelo=modelo,
          prefijo=cfg["prefijo"],
          lote=cfg["lote"],
          max_palabras=cfg["max_palabras"],
          stopwords=stopwords_config(cfg),
          metrica=cfg["metrica"],
          preprocesado=cfg["preprocesado"],
        )
        exp["plot"] = plot_embedding(items, consulta, modelo, id_sel=elegido["id"], prefijo=cfg["prefijo"])

        if len(ranking) >= 2 and selected_ids:
          primero = ranking[0]
          segundo = ranking[1]
          diferencia = primero["score"] - segundo["score"]
          exp["comparacion"] = {
            "mejor": primero,
            "segundo": segundo,
            "diferencia": diferencia,
            "mejor_comunes": elementos_en_comun(primero["texto"], consulta, stopwords=stopwords_config(cfg), preprocesado=cfg["preprocesado"]),
            "segundo_comunes": elementos_en_comun(segundo["texto"], consulta, stopwords=stopwords_config(cfg), preprocesado=cfg["preprocesado"]),
          }
      except Exception as e:
        error = (
          f"No se pudo generar la explicación: {e}. "
          f"Verifica que Ollama esté corriendo y que exista el modelo "
          f"(ollama pull {modelo})."
        )

    return render_template_string(
        PLANTILLA, vista="explicacion", items=items, consulta=consulta,
      id_sel=id_sel, exp=exp, error=error, max_palabras=cfg["max_palabras"],
        modelo=modelo, modelos=modelos, selected_ids=selected_ids,
        metrica=cfg["metrica"],
    )


# ---------- Interfaz ----------
PLANTILLA = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gestor de textos</title>
<style>
  :root {
    --fondo: #eef1ee; --papel: #fbfcfb; --tinta: #1c2321; --suave: #5d6a66;
    --linea: #cfd8d3; --acento: #0f6b5c; --acento-osc: #0a4f44; --peligro: #b3372b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--fondo); color: var(--tinta);
    font: 16px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif;
  }
  main { max-width: 46rem; margin: 0 auto; padding: 1.5rem 1rem 4rem; }
  h2 { font-size: 1.05rem; margin: 2rem 0 .75rem; }

  /* Tabs */
  nav { display: flex; gap: .25rem; border-bottom: 1px solid var(--linea); margin-bottom: 1.25rem; flex-wrap: wrap; }
  nav a {
    padding: .6rem 1.1rem; text-decoration: none; color: var(--suave);
    border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0;
    margin-bottom: -1px; font-weight: 600;
  }
  nav a:hover { color: var(--tinta); background: #e4e9e6; }
  nav a[aria-current=page] { color: var(--acento); background: var(--papel); border-color: var(--linea); }
  nav a:focus-visible { outline: 3px solid #7fc4b6; outline-offset: -3px; }

  .panel { background: var(--papel); border: 1px solid var(--linea); border-radius: 6px; padding: 1rem; }
  textarea, select, input[type=file] {
    width: 100%; font: inherit; color: inherit; background: #fff;
    border: 1px solid var(--linea); border-radius: 4px; padding: .6rem .7rem;
  }
  textarea { resize: vertical; min-height: 4.5rem; }
  textarea:focus, select:focus, button:focus-visible, input[type=file]:focus-visible, a:focus-visible {
    outline: 3px solid #7fc4b6; outline-offset: 1px;
  }
  .fila { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; margin-top: .6rem; }
  .fila > * { flex: 0 0 auto; }
  .fila .crece { flex: 1 1 12rem; }
  button {
    font: inherit; cursor: pointer; border-radius: 4px; padding: .5rem 1rem;
    border: 1px solid var(--acento); background: var(--acento); color: #fff;
  }
  button:hover { background: var(--acento-osc); }
  button.sec { background: transparent; color: var(--acento); }
  button.sec:hover { background: #e2efeb; }
  button.peligro { border-color: var(--peligro); color: var(--peligro); background: transparent; }
  button.peligro:hover { background: #f6e3e0; }
  ul { list-style: none; margin: 0; padding: 0; display: grid; gap: .6rem; }
  li { background: var(--papel); border: 1px solid var(--linea); border-radius: 6px; padding: .8rem 1rem; }
  .texto { margin: 0 0 .6rem; white-space: pre-wrap; overflow-wrap: anywhere; }
  .acciones { display: flex; gap: .5rem; }
  .acciones form { margin: 0; }
  .acciones button { padding: .3rem .8rem; font-size: .9rem; }
  .editando .vista { display: none; }
  li:not(.editando) .edicion { display: none; }
  .aviso { padding: .6rem .9rem; border-radius: 4px; margin-bottom: .75rem; border: 1px solid; }
  .aviso.ok { background: #e2efeb; border-color: #9fcbc0; }
  .aviso.error { background: #f6e3e0; border-color: #e0aaa3; }
  .vacio { color: var(--suave); margin: 0; }
  .nota { color: var(--suave); font-size: .9rem; margin: .5rem 0 0; }
  .ayuda-tuning { padding: .55rem .7rem; background: #fff3bf; border: 1px solid #e6cf72; border-radius: 4px; color: #554600; }

  .encabezado { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin: 2rem 0 .75rem; }
  .encabezado h2 { margin: 0; }
  .encabezado form { margin: 0; }
  .encabezado button { padding: .3rem .8rem; font-size: .9rem; }

  /* Resultados de búsqueda */
  li.res { display: flex; gap: 1rem; align-items: baseline; justify-content: space-between; }
  li.res .texto { margin: 0; flex: 1 1 auto; }
  .score {
    flex: 0 0 auto; min-width: 4.5rem; text-align: right;
    font-variant-numeric: tabular-nums; font-weight: 700; color: var(--acento);
  }
  a.enlace { flex: 0 0 auto; color: var(--acento); font-size: .9rem; }

  /* Explicación */
  .chips { display: flex; flex-wrap: wrap; gap: .5rem; }
  .chips li {
    padding: .25rem .7rem; border-radius: 999px; background: #e2efeb;
    border-color: #9fcbc0; font-weight: 600;
  }
  .chips li small { font-weight: 400; color: var(--suave); }
  .resaltado { line-height: 2; }
  .pal { border-radius: 3px; padding: .1rem .15rem; }
  .leyenda { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .9rem; color: var(--suave); margin: 0; }
  .muestra { display: inline-block; width: .9rem; height: .9rem; border-radius: 3px; vertical-align: -.1rem; margin-right: .3rem; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  td { padding: .35rem .5rem; border-top: 1px solid var(--linea); }
  td:last-child { text-align: right; font-weight: 700; color: var(--acento); }
</style>
</head>
<body>
<main>
  <nav aria-label="Secciones">
    <a href="{{ url_for('inicio') }}" {% if vista == 'bulk' %}aria-current="page"{% endif %}>Bulk data</a>
    <a href="{{ url_for('busqueda', q=request.args.get('q', session.get('ultima_consulta', '')), modelo=request.args.get('modelo', session.get('modelo_seleccionado', modelo if modelo else 'nomic-embed-text'))) }}" {% if vista == 'busqueda' %}aria-current="page"{% endif %}>Búsqueda</a>
    <a href="{{ url_for('explicacion', q=request.args.get('q', session.get('ultima_consulta', '')), id=request.args.get('id', session.get('ultimo_id', '')), modelo=request.args.get('modelo', session.get('modelo_seleccionado', modelo if modelo else 'nomic-embed-text'))) }}" {% if vista == 'explicacion' %}aria-current="page"{% endif %}>Explicación</a>
    <a href="{{ url_for('tuning') }}" {% if vista == 'tuning' %}aria-current="page"{% endif %}>Tuning</a>
    <a href="{{ url_for('descripcion') }}" {% if vista == 'descripcion' %}aria-current="page"{% endif %}>Descripción</a>
  </nav>

  {% with mensajes = get_flashed_messages(with_categories=true) %}
    {% for cat, msg in mensajes %}
      <div class="aviso {{ cat }}" role="status">{{ msg }}</div>
    {% endfor %}
  {% endwith %}

{% if vista == 'bulk' %}
  <form class="panel" method="post" action="{{ url_for('agregar') }}">
    <label for="texto"><strong>Insertar un texto</strong></label>
    <textarea id="texto" name="texto" placeholder="Escribe o pega el texto aquí" required></textarea>
    <div class="fila"><button type="submit">Insertar</button></div>
  </form>

  <form class="panel" method="post" action="{{ url_for('subir') }}" enctype="multipart/form-data" style="margin-top:.75rem">
    <label for="archivo"><strong>Subir un archivo .txt</strong></label>
    <div class="fila">
      <input class="crece" type="file" id="archivo" name="archivo" accept=".txt,text/plain" required>
      <select class="crece" name="modo" aria-label="Cómo separar los textos">
        <option value="lineas">Un texto por línea</option>
        <option value="parrafos">Separados por línea en blanco</option>
      </select>
      <button type="submit">Subir</button>
    </div>
    <p class="nota">Las líneas vacías se ignoran. Los textos se agregan al final de la lista.</p>
  </form>

  <div class="encabezado">
    <h2>Textos guardados ({{ items|length }})</h2>
    {% if items %}
    <form method="post" action="{{ url_for('borrar_todo') }}">
      <button type="submit" class="peligro">Eliminar todo</button>
    </form>
    {% endif %}
  </div>
  {% if items %}
  <ul>
    {% for it in items %}
    <li id="i{{ it.id }}">
      <div class="vista">
        <p class="texto">{{ it.texto }}</p>
        <div class="acciones">
          <button type="button" class="sec" onclick="alternar('{{ it.id }}')">Editar</button>
          <form method="post" action="{{ url_for('borrar', item_id=it.id) }}">
            <button type="submit" class="peligro">Borrar</button>
          </form>
        </div>
      </div>
      <form class="edicion" method="post" action="{{ url_for('editar', item_id=it.id) }}">
        <textarea name="texto" required aria-label="Editar texto">{{ it.texto }}</textarea>
        <div class="fila">
          <button type="submit">Guardar cambios</button>
          <button type="button" class="sec" onclick="alternar('{{ it.id }}')">Cancelar</button>
        </div>
      </form>
    </li>
    {% endfor %}
  </ul>
  {% else %}
    <p class="vacio">Aún no hay textos. Inserta uno arriba o sube un archivo .txt.</p>
  {% endif %}

{% elif vista == 'tuning' %}
  <div class="panel">
    <h2>Configuración de tuning</h2>
    <form method="post" action="{{ url_for('tuning') }}">
      <div style="display:grid; gap:.9rem;">
        <div>
          <label for="prefijo"><strong>Prefijo</strong></label>
          <input id="prefijo" name="prefijo" value="{{ config.prefijo }}">
          <p class="nota ayuda-tuning">Texto que se antepone a la consulta y a cada documento antes de crear sus embeddings. El valor actual es “clustering: ”; usa el prefijo recomendado por el modelo para la tarea. Puede orientar al modelo y mejorar los vectores si admite instrucciones. Esta app aplica el mismo prefijo a consultas y documentos; no hay una lista universal de valores.</p>
        </div>

        <div>
          <label for="lote"><strong>Lote</strong></label>
          <input id="lote" name="lote" type="number" min="1" max="256" value="{{ config.lote }}">
          <p class="nota ayuda-tuning">Cantidad de documentos que se envían juntos a Ollama cuando hay que calcular embeddings. Admite 1–256; predeterminado: 32. Lotes pequeños consumen menos memoria, pero pueden requerir más llamadas; lotes grandes procesan más documentos por llamada y pueden acelerar el cálculo, pero consumen más memoria.</p>
        </div>

        <div>
          <label for="stopwords"><strong>Stopwords</strong></label>
          <textarea id="stopwords" name="stopwords" rows="3" placeholder="Palabras separadas por espacios o comas">{{ config.stopwords }}</textarea>
          <p class="nota ayuda-tuning">Estas palabras se ignoran al comparar coincidencias de palabras en la explicación. No se eliminan del texto enviado al modelo y no afectan los embeddings ni el ranking semántico.</p>
        </div>

        <div>
          <label for="max_palabras"><strong>Tamaño de explicación</strong></label>
          <input id="max_palabras" name="max_palabras" type="number" min="1" max="200" value="{{ config.max_palabras }}">
          <p class="nota ayuda-tuning">Máximo de palabras del documento que se analizan para explicar qué términos aportan a la coincidencia. Admite 1–200; predeterminado: 40. Un límite bajo genera una explicación más breve y rápida; uno alto revisa más términos y puede tardar más. No cambia el ranking.</p>
        </div>

        <div>
          <label for="metrica"><strong>Métrica de distancia</strong></label>
          <select id="metrica" name="metrica" aria-describedby="metrica-ayuda">
            <option value="coseno" data-ayuda="Compara la dirección de los embeddings. Un valor más cercano a 1 indica mayor similitud." {% if config.metrica == 'coseno' %}selected{% endif %}>Coseno</option>
            <option value="euclidiana" data-ayuda="Mide la distancia en línea recta entre los vectores. Una distancia menor indica mayor similitud." {% if config.metrica == 'euclidiana' %}selected{% endif %}>Euclidiana</option>
            <option value="manhattan" data-ayuda="Suma las diferencias absolutas de cada dimensión, como contar cuadras en una cuadrícula. Una distancia menor indica mayor similitud." {% if config.metrica == 'manhattan' %}selected{% endif %}>Manhattan</option>
          </select>
          <p class="nota ayuda-tuning" id="metrica-ayuda" aria-live="polite">
            {% if config.metrica == 'manhattan' %}Suma las diferencias absolutas de cada dimensión, como contar cuadras en una cuadrícula. Una distancia menor indica mayor similitud.
            {% elif config.metrica == 'euclidiana' %}Mide la distancia en línea recta entre los vectores. Una distancia menor indica mayor similitud.
            {% else %}Compara la dirección de los embeddings. Un valor más cercano a 1 indica mayor similitud.{% endif %}
          </p>
        </div>

        <div>
          <label for="preprocesado"><strong>Preprocesado del texto</strong></label>
          <select id="preprocesado" name="preprocesado">
            <option value="normal" {% if config.preprocesado == 'normal' %}selected{% endif %}>Normal</option>
            <option value="sin_acentos" {% if config.preprocesado == 'sin_acentos' %}selected{% endif %}>Sin acentos</option>
            <option value="sin_puntuacion" {% if config.preprocesado == 'sin_puntuacion' %}selected{% endif %}>Sin puntuación</option>
            <option value="sin_acentos_ni_puntuacion" {% if config.preprocesado == 'sin_acentos_ni_puntuacion' %}selected{% endif %}>Sin acentos ni puntuación</option>
          </select>
          <p class="nota ayuda-tuning">Reglas sencillas para normalizar las palabras antes de buscar coincidencias en la explicación: “Normal” pasa a minúsculas; las otras opciones también quitan acentos, puntuación o ambos. Puede ayudar a reconocer variantes de escritura. En esta versión solo afecta la explicación léxica; no modifica el texto enviado a Ollama ni los embeddings o su ranking.</p>
        </div>

        <button type="submit">Guardar ajustes</button>
      </div>
    </form>
  </div>

{% elif vista == 'descripcion' %}
  <div class="panel">
    <h2>{{ descripcion_proyecto.titulo }}</h2>
    <p>{{ descripcion_proyecto.texto }}</p>
    <ul>
      {% for item in descripcion_proyecto.puntos %}
      <li>{{ item }}</li>
      {% endfor %}
    </ul>
    <p class="nota">Este proyecto enseña cómo se construye un sistema de búsqueda semántica con embeddings, similitud coseno y visualización del espacio vectorial.</p>
  </div>

{% elif vista == 'busqueda' %}
  <form class="panel" method="get" action="{{ url_for('busqueda') }}">
    <label for="q"><strong>Texto a comparar</strong></label>
    <textarea id="q" name="q" placeholder="Escribe el texto que quieres comparar contra los guardados" required>{{ consulta }}</textarea>
    <div class="fila">
      <select name="modelo" aria-label="Modelo de embeddings de Ollama">
        {% for m in modelos %}
        <option value="{{ m }}" {% if m == modelo %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>
      <button type="submit">Calcular similitud</button>
    </div>
  </form>

  {% if error %}
    <div class="aviso error" role="alert" style="margin-top:1rem">{{ error }}</div>
  {% endif %}

  {% if not items %}
    <h2>Resultados</h2>
    <p class="vacio">No hay textos guardados. Agrégalos primero en la pestaña Bulk data.</p>
  {% elif consulta and resultados %}
    <h2>Similitud de {{ metrica }} con “{{ consulta }}”</h2>
    <form method="get" action="{{ url_for('explicacion') }}">
      <input type="hidden" name="q" value="{{ consulta }}">
      <input type="hidden" name="modelo" value="{{ modelo }}">
      <div class="fila" style="margin-bottom:.75rem">
        <button type="submit" class="sec">Explicar seleccionados</button>
      </div>
      <ul>
        {% for r in resultados %}
        <li class="res">
          <label style="display:flex; align-items:center; gap:.75rem; width:100%;">
            <input type="checkbox" name="ids" value="{{ r.id }}">
            <p class="texto" style="margin:0; flex:1">{{ r.texto }}</p>
            <span class="score" title="Similitud de {{ metrica }}">{{ '%.4f'|format(r.score) }}</span>
            <a class="enlace" href="{{ url_for('explicacion', q=consulta, id=r.id, modelo=modelo) }}">Explicar</a>
          </label>
        </li>
        {% endfor %}
      </ul>
    </form>
    {% if metrica == 'coseno' %}
    <p class="nota">Ordenado de mayor a menor similitud: en coseno, valores más altos indican mayor parecido. Este modelo da puntajes altos incluso entre textos sin relación, así que compara los valores entre sí.</p>
    {% else %}
    <p class="nota">Ordenado por cercanía: en {{ metrica }}, una distancia menor indica mayor similitud. En esta vista se convierten los valores a una escala comparable para mantener el orden de mayor a menor cercanía.</p>
    {% endif %}
  {% elif not consulta %}
    <h2>Textos disponibles ({{ items|length }})</h2>
    {% if items %}
      <ul>
        {% for it in items %}
        <li>
          <p class="texto">{{ it.texto }}</p>
        </li>
        {% endfor %}
      </ul>
    {% else %}
      <p class="vacio">No hay textos guardados. Agrégalos primero en la pestaña Bulk data.</p>
    {% endif %}
    <p class="nota">Escribe un texto arriba para ver su similitud con cada uno de los {{ items|length }} guardados.</p>
  {% endif %}

{% else %}
  <form class="panel" method="get" action="{{ url_for('explicacion') }}">
    <label for="q"><strong>Texto a comparar</strong></label>
    <textarea id="q" name="q" placeholder="Escribe el texto de la consulta" required>{{ consulta }}</textarea>
    <div class="fila">
      <select name="modelo" aria-label="Modelo de embeddings de Ollama">
        {% for m in modelos %}
        <option value="{{ m }}" {% if m == modelo %}selected{% endif %}>{{ m }}</option>
        {% endfor %}
      </select>
      <select class="crece" name="id" aria-label="Texto a explicar">
        <option value="">Texto con mayor similitud</option>
        {% for it in items %}
        <option value="{{ it.id }}" {% if it.id == id_sel %}selected{% endif %}>{{ it.texto[:70] }}</option>
        {% endfor %}
      </select>
      <button type="submit">Explicar</button>
    </div>
  </form>

  {% if error %}
    <div class="aviso error" role="alert" style="margin-top:1rem">{{ error }}</div>
  {% endif %}

  {% if not items %}
    <h2>Explicación</h2>
    <p class="vacio">No hay textos guardados. Agrégalos primero en la pestaña Bulk data.</p>
  {% elif not consulta %}
    <h2>Explicación</h2>
    <p class="vacio">Escribe una consulta y elige un texto para ver por qué se parece. También puedes entrar desde Explicar en los resultados de la búsqueda.</p>
  {% elif exp %}
    <h2>Por qué se seleccionó este texto</h2>
    <div class="panel">
      <p class="texto"><strong>{{ exp.texto }}</strong></p>
      <p style="margin:0">{{ exp.resumen }}</p>
    </div>

    {% if exp.plot %}
    <h2>Representación del embedding</h2>
    <div class="panel">
      <svg viewBox="0 0 100 100" width="100%" height="240" aria-label="Proyección PCA de los embeddings" style="display:block;border:1px solid #cfd8d3;border-radius:6px;background:linear-gradient(#f7faf8,#edf3f1);">
        <g stroke="#d7e1dd" stroke-width="0.8">
          {% for i in range(1, 11) %}
          <line x1="{{ 12 + i * 7.2 }}" y1="12" x2="{{ 12 + i * 7.2 }}" y2="88" />
          <line x1="12" y1="{{ 12 + i * 7.2 }}" x2="88" y2="{{ 12 + i * 7.2 }}" />
          {% endfor %}
        </g>
        <line x1="12" y1="12" x2="12" y2="88" stroke="#3d4a45" stroke-width="0.8" />
        <line x1="12" y1="88" x2="88" y2="88" stroke="#3d4a45" stroke-width="0.8" />
        <text x="12" y="10" font-size="3.5" fill="#37413f">Y</text>
        <text x="88" y="93" font-size="3.5" fill="#37413f">X</text>
        {% for punto in exp.plot.puntos %}
        <circle cx="{{ punto.x }}" cy="{{ punto.y }}" r="{% if punto.selected %}4.5{% else %}3{% endif %}" fill="{% if punto.selected %}#0f6b5c{% else %}#7fc4b6{% endif %}" opacity="{% if punto.selected %}1{% else %}0.8{% endif %}"></circle>
        {% endfor %}
        <circle cx="{{ exp.plot.query.x }}" cy="{{ exp.plot.query.y }}" r="4.8" fill="#b3372b" stroke="#fff" stroke-width="1"></circle>
      </svg>
      <p class="leyenda">
        <span><span class="muestra" style="background:#0f6b5c"></span>Texto seleccionado</span>
        <span><span class="muestra" style="background:#7fc4b6"></span>Otros textos</span>
        <span><span class="muestra" style="background:#b3372b"></span>Consulta</span>
      </p>
    </div>
    {% endif %}

    {% if exp.comparacion %}
    <h2>Comparación entre seleccionados</h2>
    <div class="panel">
      <p style="margin-top:0"><strong>{{ exp.comparacion.mejor.texto }}</strong> quedó mejor rankeado que <strong>{{ exp.comparacion.segundo.texto }}</strong> por una diferencia de <strong>{{ '%.4f'|format(exp.comparacion.diferencia) }}</strong>.</p>
      <p><strong>Por qué ganó:</strong>
        {% set mejor_comunes = exp.comparacion.mejor_comunes %}
        {% set segundo_comunes = exp.comparacion.segundo_comunes %}
        {% if mejor_comunes %}
          {{ mejor_comunes|length }} elementos coinciden con la consulta en el ganador: {{ mejor_comunes|map(attribute='texto')|join(', ') }}.
        {% else %}
          El ganador no comparte palabras directas, pero su vector está más cerca semánticamente de la consulta.
        {% endif %}
      </p>
      <p><strong>Qué hizo más débil al segundo:</strong>
        {% if segundo_comunes %}
          {{ segundo_comunes|length }} elementos coinciden, pero menos relevantes o con menor vínculo semántico: {{ segundo_comunes|map(attribute='texto')|join(', ') }}.
        {% else %}
          El segundo texto no comparte palabras clave y por eso quedó más abajo en el ranking.
        {% endif %}
      </p>
    </div>
    {% endif %}

    <h2>Elementos que coinciden</h2>
    {% if exp.comunes %}
    <ul class="chips">
      {% for c in exp.comunes %}
      <li>{{ c.texto }}{% if c.texto|lower != c.consulta|lower %} <small>(consulta: {{ c.consulta }})</small>{% endif %}</li>
      {% endfor %}
    </ul>
    <p class="nota">Palabras con significado propio que aparecen en ambos textos, incluidas variantes como corre y corren.</p>
    {% else %}
    <p class="vacio">Ninguna palabra coincide. Si la similitud es alta, viene de palabras de significado parecido, como se ve abajo.</p>
    {% endif %}

    <h2>Aporte de cada palabra</h2>
    {% if exp.una_palabra %}
      <p class="vacio">El texto tiene una sola palabra, no hay nada que comparar por partes.</p>
    {% else %}
    <div class="panel">
      <p class="texto resaltado">{% for s in exp.segmentos %}{% if s.tip %}<span class="pal" style="{{ s.estilo }}" title="{{ s.tip }}">{{ s.t }}</span>{% else %}{{ s.t }}{% endif %}{% endfor %}</p>
      <p class="leyenda">
        <span><span class="muestra" style="background: rgba(15,107,92,.6)"></span>Sube la similitud</span>
        <span><span class="muestra" style="background: rgba(179,55,43,.6)"></span>La baja</span>
      </p>
    </div>
    {% if exp.top %}
    <table style="margin-top:.75rem">
      {% for palabra, delta in exp.top %}
      <tr><td>{{ palabra }}</td><td>{{ '%+.4f'|format(delta) }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
    <p class="nota">Aporte = cuánto baja la similitud al quitar esa palabra del texto. Los embeddings no comparan palabra por palabra, así que esto es una aproximación.{% if exp.recortado %} Se analizaron las primeras {{ max_palabras }} palabras.{% endif %}</p>
    {% endif %}
  {% endif %}
{% endif %}
</main>
<script>
  function alternar(id) {
    const li = document.getElementById('i' + id);
    li.classList.toggle('editando');
    if (li.classList.contains('editando')) li.querySelector('textarea').focus();
  }

  const selectorMetrica = document.getElementById('metrica');
  const ayudaMetrica = document.getElementById('metrica-ayuda');
  if (selectorMetrica && ayudaMetrica) {
    const actualizarAyudaMetrica = () => {
      ayudaMetrica.textContent = selectorMetrica.selectedOptions[0].dataset.ayuda;
    };
    selectorMetrica.addEventListener('change', actualizarAyudaMetrica);
    actualizarAyudaMetrica();
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # Se usa 5001 porque 5000 suele estar ocupado por el servicio de AirPlay en macOS.
    app.run(host="0.0.0.0", port=5001, debug=False)