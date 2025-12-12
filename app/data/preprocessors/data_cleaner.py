import pandas as pd
import json

def clean_data(raw_data):
    """
    Implementar la limpieza de datos aquí
    Maneja diccionarios en las columnas antes de eliminar duplicados
    """
    if raw_data.empty:
        return raw_data
    
    # Crear una copia para no modificar los datos originales
    cleaned_data = raw_data.copy()
    
    # Convertir columnas con diccionarios a strings para poder usar drop_duplicates
    dict_columns = []
    for col in cleaned_data.columns:
        if cleaned_data[col].apply(lambda x: isinstance(x, dict)).any():
            dict_columns.append(col)
            # Convertir diccionarios a strings JSON para comparación
            cleaned_data[col + '_str'] = cleaned_data[col].apply(
                lambda x: convert_dict_to_json(x) if isinstance(x, dict) else str(x)
            )
    
    # Eliminar filas con valores nulos en columnas críticas
    critical_columns = ['driver', 'best_lap_time', 'clean_air_pace']
    existing_critical = [col for col in critical_columns if col in cleaned_data.columns]
    if existing_critical:
        cleaned_data = cleaned_data.dropna(subset=existing_critical)
    
    # Eliminar duplicados usando las columnas string para diccionarios
    if dict_columns:
        # Usar las columnas string para detectar duplicados
        str_columns = [col + '_str' for col in dict_columns]
        other_columns = [col for col in cleaned_data.columns if col not in dict_columns and not col.endswith('_str')]
        subset_for_duplicates = other_columns + str_columns
        cleaned_data = cleaned_data.drop_duplicates(subset=subset_for_duplicates)
        # Eliminar las columnas auxiliares
        cleaned_data = cleaned_data.drop(columns=str_columns)
    else:
        cleaned_data = cleaned_data.drop_duplicates()
    
    return cleaned_data

def convert_dict_to_json(d):
    """
    Convierte un diccionario a JSON manejando tipos especiales como Timedelta
    """
    if not isinstance(d, dict):
        return str(d)
    
    # Crear una copia del diccionario con valores convertidos
    converted_dict = {}
    for key, value in d.items():
        if pd.isna(value):
            converted_dict[key] = None
        elif hasattr(value, 'total_seconds'):  # Es un Timedelta
            converted_dict[key] = value.total_seconds()
        elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
            converted_dict[key] = str(value)
        else:
            converted_dict[key] = value
    
    try:
        return json.dumps(converted_dict, sort_keys=True)
    except (TypeError, ValueError):
        return str(converted_dict)

def prepare_data(cleaned_data):
    """
    Implementar la preparación de datos aquí
    Por ejemplo, convertir tipos de datos o normalizar características
    """
    if cleaned_data.empty:
        return cleaned_data
        
    prepared_data = cleaned_data.copy()
    
    # Convertir best_lap_time a float si no lo es
    if 'best_lap_time' in prepared_data.columns:
        prepared_data['best_lap_time'] = pd.to_numeric(prepared_data['best_lap_time'], errors='coerce')
    
    # Convertir clean_air_pace a float si no lo es
    if 'clean_air_pace' in prepared_data.columns:
        prepared_data['clean_air_pace'] = pd.to_numeric(prepared_data['clean_air_pace'], errors='coerce')
    
    # Convertir sector_times a formato numérico si es posible
    if 'sector_times' in prepared_data.columns:
        prepared_data['sector_times'] = prepared_data['sector_times'].apply(convert_sector_times)
    
    return prepared_data

def convert_sector_times(sector_dict):
    """
    Convierte los tiempos de sector a segundos
    """
    if not isinstance(sector_dict, dict):
        return sector_dict
    
    converted = {}
    for sector, time_value in sector_dict.items():
        if pd.isna(time_value):
            converted[sector] = None
        elif hasattr(time_value, 'total_seconds'):
            converted[sector] = time_value.total_seconds()
        else:
            converted[sector] = time_value
    
    return converted

def filter_relevant_features(prepared_data):
    """
    Filtrar las características relevantes para el modelo
    """
    if prepared_data.empty:
        return prepared_data
        
    # Verificar qué columnas existen antes de filtrar
    available_columns = prepared_data.columns.tolist()
    relevant_features = ['driver', 'best_lap_time', 'clean_air_pace']
    
    # Solo incluir columnas que existan
    existing_features = [col for col in relevant_features if col in available_columns]
    
    # Incluir sector_times si existe
    if 'sector_times' in available_columns:
        existing_features.append('sector_times')
    
    return prepared_data[existing_features]