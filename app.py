# -*- coding: utf-8 -*-
"""LogiSense | Resumen operativo y recomendaciones de transporte."""
from __future__ import annotations

import io
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="LogiSense | Optimización de fletes", page_icon="🚚", layout="wide")

MESES = [
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
]
CAPACIDADES = {
    "TRAILER": {"tarimas": 30, "kgs": 24000},
    "TORTON": {"tarimas": 12, "kgs": 12000},
    "RABON": {"tarimas": 8, "kgs": 8000},
    "CAMIONETA 3.5": {"tarimas": 4, "kgs": 3500},
    "CAMIONETA 1.5": {"tarimas": 2, "kgs": 1500},
}
TIPOS_UNIDAD = list(CAPACIDADES)
COLUMNAS_REQUERIDAS = [
    "CLIENTE", "LOCALIDAD", "IMPORTE FACTURADO SIN IVA",
    "TARIMAS TOTALES POR VIAJE", "FLETE FACTURA", "INDICE VIAJES",
    "TIPO DE TRANSPORTE", "MES FACTURA", "KG MOVIDOS",
]


def normalizar_texto(valor: object) -> str:
    """Normaliza para comparar sin sensibilidad a acentos, espacios o mayúsculas."""
    if pd.isna(valor):
        return ""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    texto = re.sub(r"\s+", " ", texto.strip().upper())
    return texto


def normalizar_columna(nombre: object) -> str:
    return normalizar_texto(nombre).replace(".", "").replace("_", " ")


def numero_limpio(serie: pd.Series) -> pd.Series:
    texto = serie.astype(str).str.strip()
    texto = texto.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    texto = texto.str.replace(r"[$,%\s]", "", regex=True)
    # Si hay coma y punto, se asume que la coma es separador de miles.
    ambos = texto.str.contains(",", na=False) & texto.str.contains(r"\.", na=False)
    texto.loc[ambos] = texto.loc[ambos].str.replace(",", "", regex=False)
    # Si solo hay coma, se considera decimal.
    solo_coma = texto.str.contains(",", na=False) & ~texto.str.contains(r"\.", na=False)
    texto.loc[solo_coma] = texto.loc[solo_coma].str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce").fillna(0)


def estandarizar_unidad(valor: object) -> str:
    texto = normalizar_texto(valor)
    equivalencias = {
        "CAMIONETA 1.5": "CAMIONETA 1.5", "CAMIONETA 15": "CAMIONETA 1.5",
        "1.5 TON": "CAMIONETA 1.5", "1.5T": "CAMIONETA 1.5",
        "CAMIONETA 3.5": "CAMIONETA 3.5", "CAMIONETA 35": "CAMIONETA 3.5",
        "3.5 TON": "CAMIONETA 3.5", "3.5T": "CAMIONETA 3.5",
        "RABON": "RABON", "RABÓN": "RABON",
        "TORTON": "TORTON", "TORTÓN": "TORTON",
        "TRAILER": "TRAILER", "TRÁILER": "TRAILER", "TRACTOCAMION": "TRAILER",
    }
    return equivalencias.get(texto, texto)


def mes_estandar(valor: object) -> str:
    texto = normalizar_texto(valor)
    return texto if texto in MESES else texto


def encontrar_columna(columnas: pd.Index, objetivo: str) -> str | None:
    objetivo_norm = normalizar_columna(objetivo)
    for columna in columnas:
        if normalizar_columna(columna) == objetivo_norm:
            return columna
    return None


@st.cache_data(show_spinner=False)
def cargar_operacion(archivo_bytes: bytes) -> tuple[pd.DataFrame, list[str], list[str]]:
    libro = pd.ExcelFile(io.BytesIO(archivo_bytes))
    hojas_validas = [hoja for hoja in libro.sheet_names if normalizar_texto(hoja) in MESES]
    partes, errores = [], []

    for hoja in hojas_validas:
        temporal = pd.read_excel(libro, sheet_name=hoja)
        temporal.columns = [str(col).strip() for col in temporal.columns]
        faltantes = [col for col in COLUMNAS_REQUERIDAS if encontrar_columna(temporal.columns, col) is None]
        if faltantes:
            errores.append(f"{hoja}: faltan {', '.join(faltantes)}")
            continue
        renombres = {
            encontrar_columna(temporal.columns, campo): campo
            for campo in COLUMNAS_REQUERIDAS
        }
        temporal = temporal.rename(columns=renombres)
        temporal["MES_ORIGEN_HOJA"] = normalizar_texto(hoja)
        partes.append(temporal)

    if not partes:
        return pd.DataFrame(), hojas_validas, errores

    datos = pd.concat(partes, ignore_index=True)
    for columna in ["IMPORTE FACTURADO SIN IVA", "TARIMAS TOTALES POR VIAJE", "FLETE FACTURA", "KG MOVIDOS"]:
        datos[columna] = numero_limpio(datos[columna])
    datos["CLIENTE"] = datos["CLIENTE"].fillna("").astype(str).str.strip()
    datos["Destino"] = datos["LOCALIDAD"].fillna("").astype(str).str.strip()
    datos["DESTINO_MATCH"] = datos["Destino"].map(normalizar_texto)
    datos["TIPO_UNIDAD"] = datos["TIPO DE TRANSPORTE"].map(estandarizar_unidad)
    datos["MES"] = datos["MES FACTURA"].map(mes_estandar)
    datos.loc[~datos["MES"].isin(MESES), "MES"] = datos["MES_ORIGEN_HOJA"]
    datos["INDICE_LIMPIO"] = datos["INDICE VIAJES"].fillna("").astype(str).str.strip()
    datos["ID_VIAJE"] = np.where(
        datos["INDICE_LIMPIO"].isin(["", "nan", "None"]),
        "FILA_" + datos.index.astype(str),
        datos["INDICE_LIMPIO"],
    )
    return datos, hojas_validas, errores


@st.cache_data(show_spinner=False)
def cargar_tarifas(archivo_bytes: bytes) -> tuple[pd.DataFrame, str | None]:
    try:
        libro = pd.ExcelFile(io.BytesIO(archivo_bytes))
        tarifa = pd.read_excel(libro, sheet_name=libro.sheet_names[0])
    except Exception as exc:
        return pd.DataFrame(), f"No se pudo leer el archivo de tarifas: {exc}"

    tarifa.columns = [str(col).strip() for col in tarifa.columns]
    requeridas = ["TRANSPORTISTA", "UNIDAD", "COSTO DE FLETE", "DESTINO"]
    mapeo = {encontrar_columna(tarifa.columns, campo): campo for campo in requeridas}
    if any(col is None for col in mapeo):
        # Alternativa solicitada: columnas A-D aunque no tengan encabezados correctos.
        if tarifa.shape[1] < 4:
            return pd.DataFrame(), "El tarifario requiere al menos cuatro columnas."
        tarifa = tarifa.iloc[:, :4].copy()
        tarifa.columns = requeridas
    else:
        tarifa = tarifa.rename(columns=mapeo)

    tarifa = tarifa[requeridas].copy()
    tarifa["TRANSPORTISTA"] = tarifa["TRANSPORTISTA"].fillna("").astype(str).str.strip()
    tarifa["UNIDAD"] = tarifa["UNIDAD"].map(estandarizar_unidad)
    tarifa["DESTINO_MATCH"] = tarifa["DESTINO"].map(normalizar_texto)
    tarifa["COSTO DE FLETE"] = numero_limpio(tarifa["COSTO DE FLETE"])
    tarifa = tarifa[
        tarifa["UNIDAD"].isin(TIPOS_UNIDAD)
        & tarifa["DESTINO_MATCH"].ne("")
        & tarifa["COSTO DE FLETE"].gt(0)
    ].copy()
    if tarifa.empty:
        return tarifa, "No se encontraron tarifas válidas para las unidades configuradas."
    return tarifa, None


def resumir(datos: pd.DataFrame) -> pd.DataFrame:
    claves = ["CLIENTE", "Destino", "DESTINO_MATCH", "MES"]
    agregados = (
        datos.groupby(claves, dropna=False)
        .agg(
            **{
                "Cantidad de viajes": ("ID_VIAJE", "nunique"),
                "Suma facturado": ("IMPORTE FACTURADO SIN IVA", "sum"),
                "Suma tarimas": ("TARIMAS TOTALES POR VIAJE", "sum"),
                "Suma flete": ("FLETE FACTURA", "sum"),
                "KG movidos": ("KG MOVIDOS", "sum"),
            }
        )
        .reset_index()
    )
    conteos = (
        datos.assign(TIPO_UNIDAD=datos["TIPO_UNIDAD"].where(datos["TIPO_UNIDAD"].isin(TIPOS_UNIDAD)))
        .pivot_table(index=claves, columns="TIPO_UNIDAD", values="ID_VIAJE", aggfunc="size", fill_value=0)
        .reindex(columns=TIPOS_UNIDAD, fill_value=0)
        .reset_index()
    )
    resultado = agregados.merge(conteos, on=claves, how="left")
    resultado["Tarimas promedio"] = np.where(
        resultado["Cantidad de viajes"].gt(0),
        resultado["Suma tarimas"] / resultado["Cantidad de viajes"], 0,
    )
    resultado["Promedio flete"] = np.where(
        resultado["Cantidad de viajes"].gt(0),
        resultado["Suma flete"] / resultado["Cantidad de viajes"], 0,
    )
    resultado["Flete/Tarimas"] = np.where(
        resultado["Suma tarimas"].gt(0),
        resultado["Suma flete"] / resultado["Suma tarimas"], 0,
    )
    return resultado.sort_values(["CLIENTE", "Destino", "MES"]).reset_index(drop=True)


def recomendaciones(resumen: pd.DataFrame, tarifas: pd.DataFrame) -> pd.DataFrame:
    salida = []
    for _, fila in resumen.iterrows():
        disponibles = tarifas[tarifas["DESTINO_MATCH"].eq(fila["DESTINO_MATCH"])].copy()
        opciones = []
        for unidad, capacidad in CAPACIDADES.items():
            candidatos = disponibles[disponibles["UNIDAD"].eq(unidad)]
            if candidatos.empty:
                continue
            unidades_necesarias = max(
                math.ceil(fila["Suma tarimas"] / capacidad["tarimas"]) if fila["Suma tarimas"] > 0 else 1,
                math.ceil(fila["KG movidos"] / capacidad["kgs"]) if fila["KG movidos"] > 0 else 1,
            )
            mejor = candidatos.loc[candidatos["COSTO DE FLETE"].idxmin()]
            opciones.append({
                "unidad": unidad,
                "unidades": unidades_necesarias,
                "costo_total": unidades_necesarias * mejor["COSTO DE FLETE"],
                "transportista": mejor["TRANSPORTISTA"],
            })
        opciones.sort(key=lambda opcion: opcion["costo_total"])
        base = fila.to_dict()
        if opciones:
            primera = opciones[0]
            base["Recomendación principal"] = (
                f'{primera["unidades"]} {primera["unidad"]} con {primera["transportista"]} '
                f'— costo estimado ${primera["costo_total"]:,.2f}'
            )
            if len(opciones) > 1:
                segunda = opciones[1]
                base["Alternativa si el CEDIS no acepta la primera"] = (
                    f'{segunda["unidades"]} {segunda["unidad"]} con {segunda["transportista"]} '
                    f'— costo estimado ${segunda["costo_total"]:,.2f}'
                )
            else:
                base["Alternativa si el CEDIS no acepta la primera"] = "No hay una segunda unidad tarifada para este destino."
        else:
            base["Recomendación principal"] = "Sin tarifa compatible para este destino."
            base["Alternativa si el CEDIS no acepta la primera"] = "Carga tarifas para este destino."
        salida.append(base)
    return pd.DataFrame(salida)


def excel_bytes(tablas: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for hoja, tabla in tablas.items():
            tabla.to_excel(writer, sheet_name=hoja[:31], index=False)
    return buffer.getvalue()


st.title("🚚 LogiSense | Resumen y optimización de fletes")
st.caption("Procesa únicamente hojas mensuales y recomienda la combinación tarifada más económica por cliente, destino y mes.")

with st.expander("Capacidades operativas configuradas", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [{"Unidad": unidad, "Tarimas": valores["tarimas"], "KG": valores["kgs"]} for unidad, valores in CAPACIDADES.items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

col_carga, col_tarifa = st.columns(2)
with col_carga:
    archivo_operacion = st.file_uploader("Archivo operativo Excel", type=["xlsx"])
with col_tarifa:
    archivo_tarifas = st.file_uploader(
        "Tarifario Excel opcional", type=["xlsx"],
        help="Columnas esperadas: TRANSPORTISTA, UNIDAD, COSTO DE FLETE y DESTINO.",
    )

if archivo_operacion is None:
    st.info("Carga el Excel operativo para comenzar.")
    st.stop()

with st.spinner("Leyendo hojas mensuales y preparando indicadores..."):
    datos, hojas, errores = cargar_operacion(archivo_operacion.getvalue())

if datos.empty:
    st.error("No se encontraron hojas mensuales válidas con las columnas requeridas.")
    if hojas:
        st.write("Hojas mensuales detectadas:", ", ".join(hojas))
    if errores:
        st.write("Detalle:", " | ".join(errores))
    st.stop()

if errores:
    st.warning("Se omitieron hojas con estructura incompleta: " + " | ".join(errores))

tarifas = pd.DataFrame()
if archivo_tarifas is not None:
    with st.spinner("Validando tarifario y normalizando destinos..."):
        tarifas, error_tarifa = cargar_tarifas(archivo_tarifas.getvalue())
    if error_tarifa:
        st.warning(error_tarifa)
    elif not tarifas.empty:
        st.success(f"Tarifario listo: {len(tarifas):,} tarifas válidas. El match de destino ignora acentos, mayúsculas y espacios extra.")

st.sidebar.header("Filtros")
clientes = sorted(datos["CLIENTE"].dropna().unique().tolist())
meses_disponibles = [mes for mes in MESES if mes in datos["MES"].unique()]
clientes_sel = st.sidebar.multiselect("Cliente", clientes)
meses_sel = st.sidebar.multiselect("Periodo mensual", meses_disponibles, default=meses_disponibles)
destinos = sorted(datos["Destino"].dropna().unique().tolist())
destinos_sel = st.sidebar.multiselect("Destino", destinos)

filtrado = datos.copy()
if clientes_sel:
    filtrado = filtrado[filtrado["CLIENTE"].isin(clientes_sel)]
if meses_sel:
    filtrado = filtrado[filtrado["MES"].isin(meses_sel)]
if destinos_sel:
    filtrado = filtrado[filtrado["Destino"].isin(destinos_sel)]

if filtrado.empty:
    st.warning("Los filtros no devuelven registros.")
    st.stop()

tabla_resumen = resumir(filtrado)
viajes = tabla_resumen["Cantidad de viajes"].sum()
flete = tabla_resumen["Suma flete"].sum()
tarimas = tabla_resumen["Suma tarimas"].sum()
kgs = tabla_resumen["KG movidos"].sum()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Viajes únicos", f"{viajes:,.0f}")
k2.metric("Flete facturado", f"${flete:,.2f}")
k3.metric("Tarimas movilizadas", f"{tarimas:,.0f}")
k4.metric("KG movidos", f"{kgs:,.0f}")

tab_resumen, tab_sugerencias, tab_visual = st.tabs(["📋 Resumen agrupado", "💡 Sugerencias", "📊 Visual"])

with tab_resumen:
    st.subheader("Cliente y destino por mes")
    mostrar_resumen = tabla_resumen.drop(columns=["DESTINO_MATCH"])
    st.dataframe(
        mostrar_resumen.style.format({
            "Suma facturado": "${:,.2f}", "Suma flete": "${:,.2f}",
            "Tarimas promedio": "{:,.2f}", "Promedio flete": "${:,.2f}",
            "Flete/Tarimas": "${:,.2f}", "KG movidos": "{:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

with tab_sugerencias:
    st.subheader("Recomendación de unidades basada en capacidad y tarifa")
    if tarifas.empty:
        st.info("Carga un tarifario para calcular recomendaciones y alternativas por destino.")
    else:
        tabla_sugerencias = recomendaciones(tabla_resumen, tarifas)
        columnas_sugerencia = [
            "CLIENTE", "Destino", "MES", "Suma tarimas", "KG movidos",
            "Recomendación principal", "Alternativa si el CEDIS no acepta la primera",
        ]
        st.dataframe(
            tabla_sugerencias[columnas_sugerencia].style.format({"Suma tarimas": "{:,.0f}", "KG movidos": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "La recomendación elige la alternativa de menor costo total que cubra simultáneamente tarimas y KG. "
            "Solo considera unidades que tengan tarifa cargada para el destino normalizado."
        )

with tab_visual:
    grafica = (
        tabla_resumen.groupby("MES", as_index=False)["Suma flete"].sum()
        .assign(MES=lambda tabla: pd.Categorical(tabla["MES"], categories=MESES, ordered=True))
        .sort_values("MES")
    )
    fig = px.bar(grafica, x="MES", y="Suma flete", text_auto=".2s", color_discrete_sequence=["#1f77b4"])
    fig.update_layout(title="Flete facturado por mes", xaxis_title="", yaxis_title="Flete", height=410)
    st.plotly_chart(fig, use_container_width=True)

tablas_exportar = {"Resumen": mostrar_resumen}
if not tarifas.empty:
    tablas_exportar["Sugerencias"] = recomendaciones(tabla_resumen, tarifas).drop(columns=["DESTINO_MATCH"])
st.download_button(
    "⬇️ Descargar resultados en Excel",
    data=excel_bytes(tablas_exportar),
    file_name="logisense_resultados.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
