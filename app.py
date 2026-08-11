import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Optimización de fletes", page_icon="🚚", layout="wide")

MONTH_SHEETS = {
    "ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
    "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE",
}
MONTH_ORDER = list(MONTH_SHEETS)
VEHICLE_ORDER = ["CAMIONETA 1.5", "CAMIONETA 3.5", "RABON", "TORTON", "TRAILER"]
CAPACITIES = {
    "CAMIONETA 1.5": {"tarimas": 2, "kg": 1500},
    "CAMIONETA 3.5": {"tarimas": 4, "kg": 3500},
    "RABON": {"tarimas": 8, "kg": 8000},
    "TORTON": {"tarimas": 12, "kg": 12000},
    "TRAILER": {"tarimas": 30, "kg": 24000},
}


def normalize_text(value):
    """Normaliza mayúsculas, acentos, espacios y caracteres para hacer coincidencias robustas."""
    if pd.isna(value):
        return ""
    value = str(value).strip().upper()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_unit(value):
    value = normalize_text(value)
    aliases = {
        "CAMIONETA 1.5 TON": "CAMIONETA 1.5",
        "CAMIONETA 1.5T": "CAMIONETA 1.5",
        "CAMIONETA 3.5 TON": "CAMIONETA 3.5",
        "CAMIONETA 3.5T": "CAMIONETA 3.5",
        "CAMION 3.5": "CAMIONETA 3.5",
        "RABÓN": "RABON",
        "TRÁILER": "TRAILER",
        "TRAILER 53": "TRAILER",
    }
    return aliases.get(value, value)


def money_to_number(series):
    cleaned = (
        series.astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0)


def find_column(frame, wanted_name):
    target = normalize_text(wanted_name)
    matches = [col for col in frame.columns if normalize_text(col) == target]
    return matches[0] if matches else None


def load_monthly_operations(file_bytes):
    excel = pd.ExcelFile(BytesIO(file_bytes))
    usable_sheets = [sheet for sheet in excel.sheet_names if normalize_text(sheet) in MONTH_SHEETS]
    if not usable_sheets:
        raise ValueError("No se encontraron hojas con nombres de meses como ENERO, FEBRERO o MARZO.")

    required = [
        "CLIENTE", "LOCALIDAD", "IMPORTE FACTURADO SIN IVA",
        "TARIMAS TOTALES POR VIAJE", "FLETE FACTURA", "INDICE VIAJES",
        "TIPO DE TRANSPORTE", "MES FACTURA", "KG MOVIDOS",
    ]
    monthly_frames = []
    errors = []
    for sheet in usable_sheets:
        frame = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet)
        rename_map = {}
        missing = []
        for column_name in required:
            actual = find_column(frame, column_name)
            if actual is None:
                missing.append(column_name)
            else:
                rename_map[actual] = column_name
        if missing:
            errors.append(f"{sheet}: faltan {', '.join(missing)}")
            continue
        frame = frame.rename(columns=rename_map)[required].copy()
        frame["HOJA_ORIGEN"] = sheet
        monthly_frames.append(frame)

    if not monthly_frames:
        raise ValueError("Ninguna hoja mensual contiene todas las columnas requeridas. " + " | ".join(errors))

    data = pd.concat(monthly_frames, ignore_index=True)
    for col in ["IMPORTE FACTURADO SIN IVA", "TARIMAS TOTALES POR VIAJE", "FLETE FACTURA", "KG MOVIDOS"]:
        data[col] = money_to_number(data[col])
    data["CLIENTE"] = data["CLIENTE"].fillna("").astype(str).str.strip()
    data["Destino"] = data["LOCALIDAD"].fillna("").astype(str).str.strip()
    data["DESTINO_MATCH"] = data["Destino"].map(normalize_text)
    data["TIPO DE TRANSPORTE"] = data["TIPO DE TRANSPORTE"].map(normalize_unit)
    data["MES FACTURA"] = data["MES FACTURA"].fillna(data["HOJA_ORIGEN"]).map(normalize_text)
    data["MES FACTURA"] = data["MES FACTURA"].replace("", np.nan).fillna(data["HOJA_ORIGEN"].map(normalize_text))
    data["INDICE VIAJES"] = data["INDICE VIAJES"].fillna("").astype(str).str.strip()
    # Sin índice se conserva cada registro como viaje independiente.
    blank_indices = data["INDICE VIAJES"].eq("")
    data.loc[blank_indices, "INDICE VIAJES"] = "SIN_INDICE_" + data.index[blank_indices].astype(str)
    return data, usable_sheets, errors


def load_rates(file_bytes):
    raw = pd.read_excel(BytesIO(file_bytes), header=None)
    if raw.shape[1] < 4:
        raise ValueError("El archivo de tarifas debe tener al menos cuatro columnas: TRANSPORTISTA, UNIDAD, COSTO DE FLETE y DESTINO.")

    rates = raw.iloc[:, :4].copy()
    rates.columns = ["TRANSPORTISTA", "UNIDAD", "COSTO DE FLETE", "DESTINO"]
    # Elimina una posible fila de encabezados y registros vacíos.
    rates = rates[rates["UNIDAD"].map(normalize_text) != "UNIDAD"].copy()
    rates["UNIDAD"] = rates["UNIDAD"].map(normalize_unit)
    rates["DESTINO"] = rates["DESTINO"].fillna("").astype(str).str.strip()
    rates["DESTINO_MATCH"] = rates["DESTINO"].map(normalize_text)
    rates["COSTO DE FLETE"] = money_to_number(rates["COSTO DE FLETE"])
    rates = rates[
        rates["UNIDAD"].isin(VEHICLE_ORDER)
        & rates["DESTINO_MATCH"].ne("")
        & rates["COSTO DE FLETE"].gt(0)
    ]
    return rates


def summarize_operations(data):
    key_columns = ["CLIENTE", "Destino", "MES FACTURA"]
    # Un viaje es único por cliente/destino/mes/índice, aunque existan varias filas asociadas.
    trip_level = (
        data.groupby(key_columns + ["INDICE VIAJES"], dropna=False, as_index=False)
        .agg(
            **{
                "Importe facturado viaje": ("IMPORTE FACTURADO SIN IVA", "sum"),
                "Tarimas viaje": ("TARIMAS TOTALES POR VIAJE", "sum"),
                "Flete viaje": ("FLETE FACTURA", "sum"),
                "KG viaje": ("KG MOVIDOS", "sum"),
            }
        )
    )
    base = (
        trip_level.groupby(key_columns, as_index=False)
        .agg(
            **{
                "Cantidad de viajes": ("INDICE VIAJES", "nunique"),
                "Suma facturado": ("Importe facturado viaje", "sum"),
                "Suma tarimas": ("Tarimas viaje", "sum"),
                "Suma flete": ("Flete viaje", "sum"),
                "KG movidos": ("KG viaje", "sum"),
            }
        )
    )
    base["Tarimas promedio"] = base["Suma tarimas"] / base["Cantidad de viajes"].replace(0, np.nan)
    base["Promedio flete"] = base["Suma flete"] / base["Cantidad de viajes"].replace(0, np.nan)
    base["Flete/Tarimas"] = base["Suma flete"] / base["Suma tarimas"].replace(0, np.nan)

    transport_counts = (
        data[data["TIPO DE TRANSPORTE"].isin(VEHICLE_ORDER)]
        .groupby(key_columns + ["TIPO DE TRANSPORTE"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VEHICLE_ORDER, fill_value=0)
        .reset_index()
    )
    result = base.merge(transport_counts, on=key_columns, how="left")
    for unit in VEHICLE_ORDER:
        result[unit] = result[unit].fillna(0).astype(int)
    return result


def recommendation_for_group(group_row, rates):
    destination_key = group_row["Destino_match"]
    destination_rates = rates[rates["DESTINO_MATCH"] == destination_key].copy()
    if destination_rates.empty:
        return "Sin tarifa coincidente para este destino. Revise el nombre del destino en el tarifario.", ""

    total_tarimas = float(group_row["Suma tarimas"])
    total_kg = float(group_row["KG movidos"])
    current_cost = float(group_row["Suma flete"])
    candidates = []
    for unit in VEHICLE_ORDER:
        unit_rates = destination_rates[destination_rates["UNIDAD"] == unit]
        if unit_rates.empty:
            continue
        capacity = CAPACITIES[unit]
        units_needed = max(
            int(np.ceil(total_tarimas / capacity["tarimas"])) if total_tarimas > 0 else 1,
            int(np.ceil(total_kg / capacity["kg"])) if total_kg > 0 else 1,
        )
        best_rate = unit_rates.loc[unit_rates["COSTO DE FLETE"].idxmin()]
        estimated_cost = units_needed * best_rate["COSTO DE FLETE"]
        candidates.append(
            {
                "Unidad": unit,
                "Unidades requeridas": units_needed,
                "Costo estimado": estimated_cost,
                "Transportista": best_rate["TRANSPORTISTA"],
            }
        )
    if not candidates:
        return "No hay unidades válidas con tarifa para este destino.", ""

    candidate_df = pd.DataFrame(candidates).sort_values(["Costo estimado", "Unidades requeridas"])
    first = candidate_df.iloc[0]
    text = (
        f"Recomendación principal: {int(first['Unidades requeridas'])} {first['Unidad']} "
        f"con {first['Transportista']} — costo estimado ${first['Costo estimado']:,.2f}."
    )
    savings = current_cost - first["Costo estimado"]
    if current_cost > 0:
        text += f" Frente al flete facturado (${current_cost:,.2f}), la variación estimada es ${savings:,.2f}."
    if len(candidate_df) > 1:
        second = candidate_df.iloc[1]
        text += (
            f" Alternativa: {int(second['Unidades requeridas'])} {second['Unidad']} "
            f"con {second['Transportista']} — ${second['Costo estimado']:,.2f}."
        )
    return text, candidate_df


def currency_columns(frame):
    money_columns = ["Suma facturado", "Suma flete", "Promedio flete", "Flete/Tarimas"]
    formatted = frame.copy()
    for col in money_columns:
        if col in formatted:
            formatted[col] = formatted[col].map(lambda val: f"${val:,.2f}" if pd.notna(val) else "-")
    for col in ["Suma tarimas", "Tarimas promedio", "KG movidos"]:
        if col in formatted:
            formatted[col] = formatted[col].map(lambda val: f"{val:,.2f}" if pd.notna(val) else "-")
    return formatted


st.title("🚚 Análisis y optimización de fletes")
st.caption("Carga las hojas mensuales de operación y un tarifario para consolidar viajes, costos y recomendaciones por cliente y destino.")

with st.sidebar:
    st.header("Archivos de entrada")
    operations_file = st.file_uploader("Excel de operaciones", type=["xlsx", "xls"])
    rates_file = st.file_uploader("Excel de tarifas", type=["xlsx", "xls"])
    st.markdown(
        "**Tarifario esperado**  \n"
        "A: Transportista · B: Unidad · C: Costo de flete · D: Destino"
    )

if not operations_file:
    st.info("Carga el Excel de operaciones para comenzar. Solo se procesarán hojas llamadas ENERO a DICIEMBRE.")
    st.stop()

try:
    operations, processed_sheets, sheet_warnings = load_monthly_operations(operations_file.getvalue())
except Exception as exc:
    st.error(f"No fue posible leer el archivo de operaciones: {exc}")
    st.stop()

summary = summarize_operations(operations)
summary["Destino_match"] = summary["Destino"].map(normalize_text)

with st.sidebar:
    st.success(f"Hojas procesadas: {', '.join(processed_sheets)}")
    if sheet_warnings:
        st.warning("Hojas omitidas: " + " | ".join(sheet_warnings))
    clients = sorted(summary["CLIENTE"].dropna().unique())
    selected_clients = st.multiselect("Cliente", clients, default=clients)
    month_choices = [month for month in MONTH_ORDER if month in summary["MES FACTURA"].unique()]
    extra_months = sorted(set(summary["MES FACTURA"].unique()) - set(month_choices))
    selected_months = st.multiselect("Período", month_choices + extra_months, default=month_choices + extra_months)

filtered = summary[
    summary["CLIENTE"].isin(selected_clients)
    & summary["MES FACTURA"].isin(selected_months)
].copy()

metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Grupos mostrados", f"{len(filtered):,}")
metric_two.metric("Viajes únicos", f"{int(filtered['Cantidad de viajes'].sum()):,}")
metric_three.metric("Tarimas", f"{filtered['Suma tarimas'].sum():,.0f}")
metric_four.metric("Flete facturado", f"${filtered['Suma flete'].sum():,.2f}")

st.subheader("Resultados agrupados")
display_columns = [
    "CLIENTE", "Destino", "MES FACTURA", "Cantidad de viajes", "Suma facturado",
    "Suma tarimas", "Suma flete", "Tarimas promedio", "Promedio flete",
    "Flete/Tarimas", "KG movidos",
] + VEHICLE_ORDER
st.dataframe(
    currency_columns(filtered[display_columns]).sort_values(["CLIENTE", "Destino", "MES FACTURA"]),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = filtered[display_columns].to_csv(index=False).encode("utf-8-sig")
st.download_button("Descargar resultados CSV", csv_bytes, "resultados_fletes.csv", "text/csv")

st.subheader("Sugerencia de unidad")
if not rates_file:
    st.info("Carga el tarifario para generar recomendaciones. El match de destino ignora acentos, mayúsculas y espacios repetidos.")
    st.stop()

try:
    rates = load_rates(rates_file.getvalue())
except Exception as exc:
    st.error(f"No fue posible leer el tarifario: {exc}")
    st.stop()

if rates.empty:
    st.warning("No se detectaron tarifas válidas. Verifica las columnas A-D, unidades y costos.")
    st.stop()

if filtered.empty:
    st.warning("No hay grupos con los filtros seleccionados.")
    st.stop()

selection_options = filtered.apply(
    lambda row: f"{row['CLIENTE']} | {row['Destino']} | {row['MES FACTURA']}", axis=1
).tolist()
selected_label = st.selectbox("Selecciona cliente, destino y mes", selection_options)
selected_position = selection_options.index(selected_label)
selected_group = filtered.iloc[selected_position]
suggestion, options = recommendation_for_group(selected_group, rates)

st.info(suggestion)
if isinstance(options, pd.DataFrame) and not options.empty:
    options_display = options.copy()
    options_display["Costo estimado"] = options_display["Costo estimado"].map(lambda val: f"${val:,.2f}")
    st.dataframe(options_display, use_container_width=True, hide_index=True)

st.caption(
    "Criterio: se calcula la cantidad mínima de unidades que cubre simultáneamente tarimas y kg. "
    "La opción principal es la de menor costo tarifado disponible para el destino; la segunda es la siguiente alternativa más económica."
)
