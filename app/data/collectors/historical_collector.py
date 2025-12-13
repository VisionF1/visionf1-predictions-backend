import fastf1
import pandas as pd
import numpy as np
from datetime import datetime

class HistoricalCollector:
    def __init__(self, circuit_name, historical_years):
        self.circuit_name = circuit_name
        self.historical_years = historical_years
        self.historical_data = []
        
    def collect_circuit_history(self):
        """Recolecta datos históricos de un circuito específico"""
        print(f"Recolectando historial del circuito: {self.circuit_name}")
        
        for year in self.historical_years:
            try:
                # Obtener todas las carreras del año
                schedule = fastf1.get_event_schedule(year)
                
                # Buscar el circuito específico
                circuit_races = schedule[schedule['EventName'].str.contains(
                    self.circuit_name.split()[0], case=False, na=False
                )]
                
                if not circuit_races.empty:
                    race_round = circuit_races.iloc[0]['RoundNumber']
                    session = fastf1.get_session(year, race_round, 'R')
                    session.load()
                    
                    race_data = self._extract_historical_data(session, year)
                    if not race_data.empty:
                        self.historical_data.append(race_data)
                        print(f"✓ Datos históricos de {year} recolectados")
                
            except Exception as e:
                print(f"Error recolectando datos de {year}: {e}")
    
    def _extract_historical_data(self, session, year):
        """Extrae datos relevantes de una sesión histórica"""
        laps = session.laps
        results = session.results
        
        historical_data = []
        
        for driver in laps['Driver'].unique():
            try:
                driver_laps = laps[laps['Driver'] == driver]
                driver_result = results[results['Abbreviation'] == driver]
                
                if driver_result.empty:
                    continue
                    
                valid_laps = driver_laps.dropna(subset=['LapTime'])
                if valid_laps.empty:
                    continue
                
                # Extraer métricas históricas
                best_lap = valid_laps['LapTime'].min()
                avg_lap = valid_laps['LapTime'].mean()
                final_position = driver_result.iloc[0]['Position'] if not pd.isna(driver_result.iloc[0]['Position']) else 20
                grid_position = driver_result.iloc[0]['GridPosition'] if not pd.isna(driver_result.iloc[0]['GridPosition']) else 20
                
                historical_data.append({
                    'year': year,
                    'driver': driver,
                    'final_position': int(final_position),
                    'grid_position': int(grid_position),
                    'best_lap_time': best_lap.total_seconds() if pd.notna(best_lap) else None,
                    'avg_lap_time': avg_lap.total_seconds() if pd.notna(avg_lap) else None,
                    'positions_gained': int(grid_position) - int(final_position),
                    'laps_completed': len(valid_laps)
                })
                
            except Exception as e:
                print(f"Error procesando piloto {driver} en {year}: {e}")
        
        return pd.DataFrame(historical_data)
    
    def get_driver_circuit_performance(self, driver):
        """Obtiene el rendimiento histórico de un piloto en el circuito"""
        if not self.historical_data:
            return None
            
        all_data = pd.concat(self.historical_data, ignore_index=True)
        driver_data = all_data[all_data['driver'] == driver]
        
        if driver_data.empty:
            return None
            
        return {
            'avg_position': driver_data['final_position'].mean(),
            'best_position': driver_data['final_position'].min(),
            'avg_grid': driver_data['grid_position'].mean(),
            'avg_positions_gained': driver_data['positions_gained'].mean(),
            'races_count': len(driver_data),
            'recent_form': driver_data.tail(3)['final_position'].mean() if len(driver_data) >= 3 else driver_data['final_position'].mean()
        }
    
    def get_historical_data(self):
        """Retorna todos los datos históricos recolectados"""
        if self.historical_data:
            return pd.concat(self.historical_data, ignore_index=True)
        return pd.DataFrame()