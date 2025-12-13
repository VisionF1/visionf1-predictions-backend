#!/usr/bin/env python3
"""
Team Mapping Utils - Funciones simples para mapear equipos históricos
Funciones de utilidad para aplicar mapeo de equipos a cualquier DataFrame
"""

import pandas as pd
from .team_mapper import F1TeamMapper


# Instancia global del mapper para reutilización
_team_mapper_instance = None


def get_team_mapper():
    """Obtiene instancia singleton del team mapper."""
    global _team_mapper_instance
    if _team_mapper_instance is None:
        _team_mapper_instance = F1TeamMapper()
    return _team_mapper_instance


def map_team_names(df, team_column='team', preserve_original=True, add_lineage_info=False):
    """
    Mapea nombres históricos de equipos a sus equivalentes actuales en 2025.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de F1
        team_column (str): Nombre de la columna que contiene los equipos (default: 'team')
        preserve_original (bool): Si conservar los nombres originales en una nueva columna (default: True)
        add_lineage_info (bool): Si agregar información de linaje del equipo (default: False)
    
    Returns:
        pandas.DataFrame: DataFrame con equipos mapeados
        
    Example:
        >>> import pandas as pd
        >>> from visionf1.core.utils.team_mapping_utils import map_team_names
        >>> 
        >>> # Tu DataFrame con datos
        >>> df = pd.read_csv('mi_dataset.csv')
        >>> 
        >>> # Mapear equipos manteniendo los originales
        >>> df_mapped = map_team_names(df)
        >>> 
        >>> # Ver transformaciones
        >>> transformations = df_mapped[df_mapped['team_original'] != df_mapped['team']]
        >>> print(transformations[['team_original', 'team']].drop_duplicates())
    """
    
    if team_column not in df.columns:
        raise ValueError(f"Columna '{team_column}' no encontrada en el DataFrame")
    
    # Crear copia del DataFrame
    df_result = df.copy()
    
    # Obtener mapper
    mapper = get_team_mapper()
    
    # Preservar nombres originales si se solicita
    if preserve_original:
        df_result[f'{team_column}_original'] = df_result[team_column].copy()
    
    # Aplicar mapeo
    df_result[team_column] = df_result[team_column].map(mapper.map_to_current)
    
    # Agregar información de linaje si se solicita
    if add_lineage_info:
        df_result[f'{team_column}_lineage'] = df_result[f'{team_column}_original'].apply(
            lambda x: mapper.map_to_current(x)
        )
        
        # Agregar flag si hubo transformación
        df_result[f'{team_column}_transformed'] = (
            df_result[f'{team_column}_original'] != df_result[team_column]
        )
    
    return df_result


def get_team_transformations(df, team_column='team'):
    """
    Obtiene un resumen de las transformaciones de equipos aplicadas.
    
    Args:
        df (pandas.DataFrame): DataFrame procesado con map_team_names()
        team_column (str): Nombre de la columna de equipos
    
    Returns:
        pandas.DataFrame: DataFrame con resumen de transformaciones
        
    Example:
        >>> transformations = get_team_transformations(df_mapped)
        >>> print(transformations)
    """
    
    original_col = f'{team_column}_original'
    
    if original_col not in df.columns:
        raise ValueError(f"DataFrame debe ser procesado primero con map_team_names()")
    
    # Encontrar registros con transformaciones
    transformed = df[df[original_col] != df[team_column]]
    
    if transformed.empty:
        return pd.DataFrame(columns=['team_original', 'team_mapped', 'records_count'])
    
    # Crear resumen
    summary = transformed.groupby([original_col, team_column]).size().reset_index(name='records_count')
    summary.columns = ['team_original', 'team_mapped', 'records_count']
    
    return summary.sort_values('records_count', ascending=False)


def get_team_lineage_summary(current_team_name=None):
    """
    Obtiene información sobre el linaje histórico de los equipos.
    
    Args:
        current_team_name (str, optional): Nombre específico de equipo actual.
                                         Si None, devuelve info de todos los equipos.
    
    Returns:
        dict: Información de linaje de equipos
        
    Example:
        >>> # Ver linaje de un equipo específico
        >>> lineage = get_team_lineage_summary('Racing Bulls')
        >>> print(lineage)
        
        >>> # Ver todos los linajes
        >>> all_lineages = get_team_lineage_summary()
        >>> print(all_lineages)
    """
    
    mapper = get_team_mapper()
    
    if current_team_name:
        # Información de un equipo específico
        historical_names = mapper.get_historical_names(current_team_name)
        evolution_info = mapper.get_team_evolution_info(current_team_name)
        
        return {
            'current_name': current_team_name,
            'historical_names': historical_names,
            'evolution_info': evolution_info
        }
    else:
        # Información de todos los equipos
        all_teams = mapper.get_all_teams_2025()
        lineages = {}
        
        for team in all_teams:
            lineages[team] = {
                'historical_names': mapper.get_historical_names(team),
                'evolution_info': mapper.get_team_evolution_info(team)
            }
        
        return lineages


def check_team_compatibility(team1, team2):
    """
    Verifica si dos nombres de equipos pertenecen al mismo linaje.
    
    Args:
        team1 (str): Primer nombre de equipo
        team2 (str): Segundo nombre de equipo
    
    Returns:
        bool: True si pertenecen al mismo linaje
        
    Example:
        >>> # ¿AlphaTauri y Racing Bulls son el mismo linaje?
        >>> is_same = check_team_compatibility('AlphaTauri', 'Racing Bulls')
        >>> print(f"Mismo linaje: {is_same}")  # True
    """
    
    mapper = get_team_mapper()
    return mapper.is_same_team_lineage(team1, team2)


def print_mapping_info():
    """
    Imprime información completa sobre el mapeo de equipos.
    Útil para entender qué transformaciones se aplicarán.
    
    Example:
        >>> from visionf1.core.utils.team_mapping_utils import print_mapping_info
        >>> print_mapping_info()
    """
    
    mapper = get_team_mapper()
    mapper.print_mapping_summary()


# Función de conveniencia para uso rápido
def quick_team_mapping(df, team_column='team'):
    """
    Función de conveniencia para mapeo rápido con configuración estándar.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de F1
        team_column (str): Nombre de la columna de equipos
    
    Returns:
        pandas.DataFrame: DataFrame con equipos mapeados y información de transformaciones
        
    Example:
        >>> # Uso más simple
        >>> df_mapped = quick_team_mapping(df)
        >>> 
        >>> # Ver si hubo transformaciones
        >>> if 'team_transformed' in df_mapped.columns:
        >>>     transformed_count = df_mapped['team_transformed'].sum()
        >>>     print(f"Registros transformados: {transformed_count}")
    """
    
    return map_team_names(
        df, 
        team_column=team_column, 
        preserve_original=True, 
        add_lineage_info=True
    )


# Función para validar que el mapeo es correcto
def validate_mapping(df_original, df_mapped, team_column='team'):
    """
    Valida que el mapeo se aplicó correctamente.
    
    Args:
        df_original (pandas.DataFrame): DataFrame original
        df_mapped (pandas.DataFrame): DataFrame después del mapeo
        team_column (str): Nombre de la columna de equipos
    
    Returns:
        dict: Reporte de validación
        
    Example:
        >>> validation = validate_mapping(df, df_mapped)
        >>> print(f"Mapeo válido: {validation['is_valid']}")
    """
    
    validation_report = {
        'is_valid': True,
        'errors': [],
        'stats': {}
    }
    
    try:
        # Verificar que no se perdieron registros
        if len(df_original) != len(df_mapped):
            validation_report['is_valid'] = False
            validation_report['errors'].append("Número de registros no coincide")
        
        # Verificar que existe columna original
        original_col = f'{team_column}_original'
        if original_col not in df_mapped.columns:
            validation_report['is_valid'] = False
            validation_report['errors'].append(f"Columna {original_col} no encontrada")
        else:
            # Estadísticas
            total_records = len(df_mapped)
            transformed_records = (df_mapped[original_col] != df_mapped[team_column]).sum()
            
            validation_report['stats'] = {
                'total_records': total_records,
                'transformed_records': transformed_records,
                'transformation_percentage': (transformed_records / total_records) * 100,
                'teams_before': df_original[team_column].nunique(),
                'teams_after': df_mapped[team_column].nunique()
            }
    
    except Exception as e:
        validation_report['is_valid'] = False
        validation_report['errors'].append(f"Error en validación: {str(e)}")
    
    return validation_report
