import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.core.features.feature_extractor import FeatureExtractor
from app.core.utils.team_mapping_utils import quick_team_mapping

class DataPreparer:
    """Prepara datos para entrenamiento de modelos ML"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
    
    def prepare_training_data(self, data):
        """Prepara datos completos para entrenamiento"""
        print("🔄 Preparando datos para entrenamiento...")
        print("🔍 DEBUG: Iniciando prepare_training_data")
        
        if data is None or data.empty:
            print("❌ No hay datos disponibles para preparar")
            return None, None, None, None, None
        
        training_data = data.copy()
        print(f"📊 Datos originales: {training_data.shape}")
        
        # 🔍 DEBUG: Mostrar columnas disponibles
        print(f"🔍 Columnas disponibles en los datos:")
        columns_list = list(training_data.columns)
        for i, col in enumerate(columns_list):
            print(f"   {i+1:2d}. {col}")
        
        # Buscar columnas relacionadas con equipos
        team_related_columns = [col for col in columns_list if 'team' in col.lower()]
        if team_related_columns:
            print(f"🏎️  Columnas relacionadas con equipos: {team_related_columns}")
        else:
            print(f"⚠️  No se encontraron columnas con 'team' en el nombre")
        
        # 🔧 APLICAR MAPEO DE EQUIPOS ANTES DE ENTRENAR
        print("🔄 Aplicando mapeo histórico de equipos...")
        if 'team' in training_data.columns:
            training_data = quick_team_mapping(training_data)
            unique_teams_after = training_data['team'].dropna().unique()
            print(f"✅ Equipos después del mapeo: {len(unique_teams_after)} equipos únicos")
            print(f"   🏎️ Equipos: {', '.join(sorted(unique_teams_after))}")
            
            # Verificar cuántos valores nulos hay
            null_teams = training_data['team'].isnull().sum()
            if null_teams > 0:
                print(f"⚠️  {null_teams} registros sin información de equipo")
                
            # 🚫 FILTRAR REGISTROS CON EQUIPOS DESCONOCIDOS
            print("🔄 Filtrando registros sin equipo válido...")
            initial_count = len(training_data)
            
            # Eliminar registros con team = null/nan o 'Unknown'
            mask_valid_team = training_data['team'].notna() & (training_data['team'] != 'Unknown')
            training_data = training_data[mask_valid_team].copy()
            
            filtered_count = len(training_data)
            removed_count = initial_count - filtered_count
            
            if removed_count > 0:
                print(f"   🗑️  {removed_count} registros eliminados (sin equipo válido)")
                print(f"   ✅ {filtered_count} registros restantes con equipos válidos")
            else:
                print(f"   ✅ Todos los registros tienen equipos válidos")
                
            # Actualizar lista de equipos únicos
            final_teams = training_data['team'].unique()
            print(f"   🏎️ Equipos finales: {len(final_teams)} - {', '.join(sorted(final_teams))}")
            
        else:
            print("⚠️ No se encontró columna 'team' en los datos")
            # Buscar alternativas
            possible_team_cols = [col for col in training_data.columns if any(word in col.lower() for word in ['equipo', 'constructor', 'team'])]
            if possible_team_cols:
                print(f"🔍 Posibles columnas de equipo alternativas: {possible_team_cols}")
            else:
                print("❌ No se encontraron columnas de equipo alternativas")
        
        # 1. Preparar características base (8 nuevas características)
        X_base, feature_names = self._prepare_base_features(training_data)
        if X_base is None:
            return None, None, None, None, None
        
        X_df = pd.DataFrame(X_base, columns=feature_names)
        
        # 2. Agregar características adicionales
        X_df = self._add_additional_features(X_df, training_data)
        
        # 3. Preparar variable objetivo ANTES de limpiar
        y = self._prepare_target_variable(training_data)
        if y is None:
            return None, None, None, None, None
        
        # 4. Asegurar misma longitud entre X y y
        min_length = min(len(X_df), len(y))
        X_df = X_df.iloc[:min_length].reset_index(drop=True)
        y = y.iloc[:min_length].reset_index(drop=True)
        training_data = training_data.iloc[:min_length].reset_index(drop=True)
        
        print(f"📏 Sincronizando datos: X{X_df.shape}, y{len(y)}")
        
        # 5. Limpiar datos
        X_clean, training_data_clean = self._clean_data(X_df, training_data, feature_names)
        
        # 6. Filtrar datos válidos
        X_final, y_final = self._filter_valid_data(X_clean, y)
        
        # MOSTRAR HEAD DEL DATASET ANTES DEL ENTRENAMIENTO (PARA DEBUG)
        #self._show_dataset_head(X_final, y_final, training_data_clean)
        
        # 7. Dividir en entrenamiento y test (preservando índices, mezclando años)
        train_indices, test_indices = train_test_split(
            range(len(X_final)), test_size=0.2, random_state=42, shuffle=True
        )
        
        X_train = X_final.iloc[train_indices].reset_index(drop=True)
        X_test = X_final.iloc[test_indices].reset_index(drop=True)
        y_train = y_final.iloc[train_indices].reset_index(drop=True)
        y_test = y_final.iloc[test_indices].reset_index(drop=True)
        
        print(f"📊 División de datos completada:")
        print(f"   🎯 Entrenamiento: {X_train.shape[0]} muestras")
        print(f"   🧪 Test: {X_test.shape[0]} muestras")
        print(f"   🔢 Total características: {X_train.shape[1]}")
        
        print("🔍 DEBUG: Antes de guardar índices")
        
        # Guardar nombres de características
        self.feature_names = list(X_train.columns)
        
        # Guardar índices para mapeo posterior
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        print(f"🔍 DEBUG: Índices guardados - train: {len(train_indices)}, test: {len(test_indices)}")
        print("🔍 DEBUG: Terminando prepare_training_data")
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def _prepare_base_features(self, data):
        """Prepara las características base principales"""
        try:
            # Lista de las características principales que necesitamos
            required_features = [
                'team_encoded',
                'team_avg_position_2024',
                'team_avg_position_2023', 
                'team_avg_position_2022',
                'grid_position',
                'quali_best_time',
                'race_best_lap_time',
                'clean_air_pace',
                # NUEVAS CARACTERÍSTICAS METEOROLÓGICAS
                'session_air_temp',
                'session_track_temp',
                'session_humidity',
                'session_rainfall'
            ]
            
            print(f"🔧 Preparando {len(required_features)} características base:")
            for i, feature in enumerate(required_features, 1):
                print(f"   {i}. {feature}")
            
            feature_data = []
            feature_names = []
            
            # 1. Team encoding
            if 'team' in data.columns:
                # Ya no deberíamos tener valores Unknown después del filtrado previo
                valid_teams = data['team'].dropna()
                if len(valid_teams) > 0:
                    team_encoded = self.label_encoder.fit_transform(valid_teams)
                    # Si hay menos registros válidos, rellenar con ceros
                    if len(team_encoded) < len(data):
                        full_encoded = np.zeros(len(data))
                        valid_mask = data['team'].notna()
                        full_encoded[valid_mask] = team_encoded
                        team_encoded = full_encoded
                    feature_data.append(team_encoded)
                    feature_names.append('team_encoded')
                    print(f"   ✅ team_encoded: {len(self.label_encoder.classes_)} equipos únicos (sin Unknown)")
                else:
                    feature_data.append(np.zeros(len(data)))
                    feature_names.append('team_encoded')
                    print(f"   ⚠️ team_encoded: no hay equipos válidos")
            else:
                feature_data.append(np.zeros(len(data)))
                feature_names.append('team_encoded')
                print(f"   ⚠️ team_encoded: usando valores por defecto")
            
            # 2-4. Team historical performance 
            if 'team' in data.columns and data['team'].notna().sum() > 0:
                # Crear performance histórico basado en el equipo (solo para equipos válidos)
                team_perf_2024 = data['team'].map(self._get_team_performance_2024).fillna(10)
                team_perf_2023 = data['team'].map(self._get_team_performance_2023).fillna(10)
                team_perf_2022 = data['team'].map(self._get_team_performance_2022).fillna(10)
                print(f"   ✅ Performance histórico calculado para {data['team'].notna().sum()} registros con equipo")
            else:
                # Valores por defecto si no hay información de equipos
                team_perf_2024 = np.full(len(data), 10.0)
                team_perf_2023 = np.full(len(data), 10.0)
                team_perf_2022 = np.full(len(data), 10.0)
                print(f"   ⚠️ Performance histórico: usando valores por defecto")
            
            feature_data.extend([team_perf_2024, team_perf_2023, team_perf_2022])
            feature_names.extend(['team_avg_position_2024', 'team_avg_position_2023', 'team_avg_position_2022'])
            
            # 5. Grid position
            grid_pos = data.get('grid_position', data.get('quali_position', np.full(len(data), 10))).fillna(10)
            feature_data.append(grid_pos)
            feature_names.append('grid_position')
            print(f"   ✅ grid_position: rango {grid_pos.min():.1f} - {grid_pos.max():.1f}")
            
            # 6. Qualifying time
            quali_time = data.get('quali_best_time', data.get('q3_time', data.get('q2_time', data.get('q1_time', np.full(len(data), 90))))).fillna(90)
            feature_data.append(quali_time)
            feature_names.append('quali_best_time')
            print(f"   ✅ quali_best_time: rango {quali_time.min():.1f} - {quali_time.max():.1f}")
            
            # 7. Race best lap time
            race_time = data.get('race_best_lap_time', data.get('best_lap_time', np.full(len(data), 90))).fillna(90)
            feature_data.append(race_time)
            feature_names.append('race_best_lap_time')
            print(f"   ✅ race_best_lap_time: rango {race_time.min():.1f} - {race_time.max():.1f}")
            
            # 8. Clean air pace
            clean_pace = data.get('clean_air_pace', np.full(len(data), 0.0)).fillna(0.0)
            feature_data.append(clean_pace)
            feature_names.append('clean_air_pace')
            print(f"   ✅ clean_air_pace: rango {clean_pace.min():.3f} - {clean_pace.max():.3f}")
            
            # 9-12. CARACTERÍSTICAS METEOROLÓGICAS
            print(f"   🌤️  Procesando condiciones meteorológicas:")
            
            # 9. Temperatura del aire
            air_temp = data.get('session_air_temp', np.full(len(data), 25.0)).fillna(25.0)  # 25°C por defecto
            feature_data.append(air_temp)
            feature_names.append('session_air_temp')
            print(f"   ✅ session_air_temp: rango {air_temp.min():.1f}°C - {air_temp.max():.1f}°C")
            
            # 10. Temperatura de la pista
            track_temp = data.get('session_track_temp', np.full(len(data), 35.0)).fillna(35.0)  # 35°C por defecto
            feature_data.append(track_temp)
            feature_names.append('session_track_temp')
            print(f"   ✅ session_track_temp: rango {track_temp.min():.1f}°C - {track_temp.max():.1f}°C")
            
            # 11. Humedad
            humidity = data.get('session_humidity', np.full(len(data), 60.0)).fillna(60.0)  # 60% por defecto
            feature_data.append(humidity)
            feature_names.append('session_humidity')
            print(f"   ✅ session_humidity: rango {humidity.min():.1f}% - {humidity.max():.1f}%")
            
            # 12. Lluvia (convertir a binario: 0=seco, 1=lluvia)
            rainfall = data.get('session_rainfall', np.full(len(data), False)).fillna(False)
            # Convertir booleano a numérico
            rainfall_numeric = rainfall.astype(int)
            feature_data.append(rainfall_numeric)
            feature_names.append('session_rainfall')
            rain_sessions = rainfall_numeric.sum()
            total_sessions = len(rainfall_numeric)
            print(f"   ✅ session_rainfall: {rain_sessions} sesiones con lluvia de {total_sessions} ({rain_sessions/total_sessions*100:.1f}%)")
            
            # Convertir a matriz numpy
            X = np.column_stack(feature_data)
            
            print(f"✅ Características base preparadas: {X.shape}")
            print(f"   🌤️  Características meteorológicas incluidas para modelar el impacto del clima")
            return X, feature_names
            
        except Exception as e:
            print(f"❌ Error preparando características base: {e}")
            return None, None
    
    def _get_team_performance_2024(self, team):
        """Retorna la posición promedio del equipo en 2024"""
        team_performance = {
            'McLaren': 2.5,
            'Ferrari': 3.2,
            'Red Bull Racing': 1.8,
            'Mercedes': 4.1,
            'Aston Martin': 6.5,
            'Alpine': 8.3,
            'Williams': 9.1,
            'Racing Bulls': 8.7,
            'Haas': 9.8,
            'Sauber': 10.2
        }
        return team_performance.get(team, 10.0)
    
    def _get_team_performance_2023(self, team):
        """Retorna la posición promedio del equipo en 2023"""
        team_performance = {
            'Red Bull Racing': 1.2,
            'Mercedes': 4.5,
            'Ferrari': 5.1,
            'McLaren': 7.2,
            'Aston Martin': 3.8,
            'Alpine': 8.9,
            'Williams': 9.5,
            'Racing Bulls': 8.1,  # Era AlphaTauri
            'Haas': 9.8,
            'Sauber': 10.1  # Era Alfa Romeo
        }
        return team_performance.get(team, 10.0)
    
    def _get_team_performance_2022(self, team):
        """Retorna la posición promedio del equipo en 2022"""
        team_performance = {
            'Red Bull Racing': 2.1,
            'Ferrari': 3.5,
            'Mercedes': 4.8,
            'McLaren': 6.5,
            'Alpine': 7.2,
            'Racing Bulls': 8.8,  # Era AlphaTauri
            'Aston Martin': 7.9,
            'Williams': 9.1,
            'Haas': 8.5,
            'Sauber': 9.7  # Era Alfa Romeo
        }
        return team_performance.get(team, 10.0)
    
    def _add_additional_features(self, X_df, training_data):
        """Agrega características adicionales al dataset"""
        print("🔧 Agregando características adicionales...")
        
        additional_features = {}
        
        # Características de tiempo normalizadas
        for time_col in ['fp1_best_time', 'fp2_best_time', 'fp3_best_time']:
            if time_col in training_data.columns:
                normalized_col = f"{time_col}_normalized"
                if normalized_col in training_data.columns:
                    additional_features[normalized_col] = training_data[normalized_col].fillna(training_data[normalized_col].median())
        
        # Características de sector
        for sector_col in ['race_sector1', 'race_sector2', 'race_sector3']:
            if sector_col in training_data.columns:
                additional_features[sector_col] = training_data[sector_col].fillna(training_data[sector_col].median())
        
        # Características de práctica libre
        if 'fp_avg_position' in training_data.columns:
            additional_features['fp_avg_position'] = training_data['fp_avg_position'].fillna(10)
        
        if 'fp_consistency' in training_data.columns:
            additional_features['fp_consistency'] = training_data['fp_consistency'].fillna(0.5)
        
        # Características de rendimiento en carrera
        if 'race_pace_rank' in training_data.columns:
            additional_features['race_pace_rank'] = training_data['race_pace_rank'].fillna(10)
        
        if 'quali_performance' in training_data.columns:
            additional_features['quali_performance'] = training_data['quali_performance'].fillna(0.5)
        
        # Características de vuelta rápida normalizada
        if 'best_lap_time_normalized' in training_data.columns:
            additional_features['best_lap_time_normalized'] = training_data['best_lap_time_normalized'].fillna(training_data['best_lap_time_normalized'].median())
        
        # Crear DataFrame con características adicionales
        if additional_features:
            additional_df = pd.DataFrame(additional_features)
            X_df = pd.concat([X_df, additional_df], axis=1)
            print(f"   ✅ {len(additional_features)} características adicionales agregadas")
        else:
            print(f"   ⚠️ No se encontraron características adicionales válidas")
        
        return X_df
    
    def _clean_data(self, X_df, training_data, feature_names):
        """Limpia datos eliminando valores infinitos y NaN"""
        print("🧹 Limpiando datos...")
        
        # Reemplazar infinitos con NaN
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        
        # Mostrar estadísticas de valores faltantes
        missing_stats = X_df.isnull().sum()
        if missing_stats.sum() > 0:
            print(f"   📊 Valores faltantes encontrados:")
            for col, missing_count in missing_stats[missing_stats > 0].items():
                print(f"      {col}: {missing_count} ({missing_count/len(X_df)*100:.1f}%)")
        
        # Rellenar valores faltantes con la mediana
        for col in X_df.columns:
            if X_df[col].isnull().sum() > 0:
                median_val = X_df[col].median()
                if pd.isna(median_val):
                    # Si la mediana también es NaN, usar un valor por defecto
                    if 'position' in col.lower():
                        median_val = 10.0
                    elif 'time' in col.lower():
                        median_val = 90.0
                    else:
                        median_val = 0.0
                # Usar método correcto para evitar FutureWarning
                X_df.loc[:, col] = X_df[col].fillna(median_val)
        
        print(f"   ✅ Datos limpios: {X_df.shape}")
        return X_df, training_data
    
    def _prepare_target_variable(self, training_data):
        """Prepara la variable objetivo (final_position)"""
        print("🎯 Preparando variable objetivo...")
        
        # Buscar columna de posición final
        target_columns = ['final_position', 'race_position', 'position']
        target_col = None
        
        for col in target_columns:
            if col in training_data.columns:
                target_col = col
                break
        
        if target_col is None:
            print("❌ No se encontró variable objetivo válida")
            return None
        
        y = training_data[target_col].copy()
        
        # Limpiar variable objetivo
        y = y.fillna(20)  # Posición por defecto para valores faltantes
        y = np.clip(y, 1, 20)  # Asegurar que esté en rango válido
        
        print(f"   ✅ Variable objetivo preparada: {target_col}")
        print(f"   📊 Rango: {y.min()} - {y.max()}")
        
        return y
    
    def _filter_valid_data(self, X_df, y):
        """Filtra datos válidos eliminando filas con problemas"""
        print("🔍 Filtrando datos válidos...")
        
        initial_count = len(X_df)
        
        # Verificar que no haya NaN restantes
        # Asegurar que X_df y y tienen el mismo índice
        X_df = X_df.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        mask_valid = ~(X_df.isnull().any(axis=1) | pd.isna(y))
        
        X_clean = X_df.loc[mask_valid].copy()
        y_clean = y.loc[mask_valid].copy()
        
        removed_count = initial_count - len(X_clean)
        if removed_count > 0:
            print(f"   ⚠️ {removed_count} filas eliminadas por datos inválidos")
        
        print(f"   ✅ Datos válidos: {len(X_clean)} muestras")
        
        return X_clean, y_clean
    
    def _show_dataset_head(self, X_df, y, original_data):
        """Muestra las primeras filas del dataset para inspección"""
        print("\n" + "="*80)
        print("📊 HEAD DEL DATASET - DATOS QUE SE USARÁN PARA ENTRENAR")
        print("="*80)
        
        # Mostrar información general
        print(f"📈 Total de muestras: {len(X_df)}")
        print(f"🔢 Total de características: {X_df.shape[1]}")
        print(f"🎯 Variable objetivo: posiciones de {y.min()} a {y.max()}")
        
        # Crear DataFrame combinado para visualización
        display_df = X_df.copy()
        display_df['target_position'] = y.values
        
        # Agregar información contextual si está disponible
        if len(original_data) >= len(X_df):
            context_cols = ['driver', 'team', 'race', 'year']
            for col in context_cols:
                if col in original_data.columns:
                    display_df[f'info_{col}'] = original_data[col].iloc[:len(X_df)].values
        
        print(f"\n📋 PRIMERAS 10 FILAS DEL DATASET:")
        print("-" * 120)
        
        # Mostrar head con formato mejorado
        head_df = display_df.head(10)
        
        # Separar características meteorológicas para destacarlas
        weather_cols = ['session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall']
        other_cols = [col for col in X_df.columns if col not in weather_cols]
        
        print("🏎️  CARACTERÍSTICAS PRINCIPALES:")
        if len(other_cols) > 0:
            for col in other_cols[:6]:  # Mostrar las primeras 6
                values = head_df[col].round(2) if head_df[col].dtype in ['float64', 'float32'] else head_df[col]
                print(f"   {col:25s}: {list(values)}")
        
        print("\n🌤️  CARACTERÍSTICAS METEOROLÓGICAS:")
        for col in weather_cols:
            if col in head_df.columns:
                values = head_df[col].round(1) if head_df[col].dtype in ['float64', 'float32'] else head_df[col]
                unit = "°C" if "temp" in col else "%" if "humidity" in col else ""
                print(f"   {col:25s}: {list(values)} {unit}")
        
        print(f"\n🎯 POSICIONES OBJETIVO:")
        print(f"   target_position        : {list(head_df['target_position'])}")
        
        # Mostrar información contextual si está disponible
        context_info = [col for col in head_df.columns if col.startswith('info_')]
        if context_info:
            print(f"\n📝 INFORMACIÓN CONTEXTUAL:")
            for col in context_info:
                clean_name = col.replace('info_', '')
                print(f"   {clean_name:25s}: {list(head_df[col])}")
        
        # Estadísticas resumidas
        print(f"\n📊 ESTADÍSTICAS RESUMIDAS:")
        print(f"   🌡️  Temperatura promedio del aire: {X_df['session_air_temp'].mean():.1f}°C")
        print(f"   🛣️  Temperatura promedio de pista: {X_df['session_track_temp'].mean():.1f}°C") 
        print(f"   💧 Humedad promedio: {X_df['session_humidity'].mean():.1f}%")
        print(f"   🌧️  Sesiones con lluvia: {X_df['session_rainfall'].sum()} de {len(X_df)} ({X_df['session_rainfall'].mean()*100:.1f}%)")
        
        print("="*80 + "\n")
        
        return display_df