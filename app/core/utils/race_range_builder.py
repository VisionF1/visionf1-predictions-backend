import fastf1
import pandas as pd

class RaceRangeBuilder:
    """Construye rangos de carreras para recolección de datos"""
    
    def build_race_range(self, config):
        """Construye el rango de carreras basado en configuración"""
        race_range = []
        years = config.get("years", [2024])
        max_races_per_year = config.get("max_races_per_year", 24)
        
        for year in years:
            races = self._get_races_for_year(year, max_races_per_year)
            race_range.extend(races)
        
        return race_range
    
    def _get_races_for_year(self, year, max_races):
        """Obtiene carreras disponibles para un año específico"""
        try:
            schedule = fastf1.get_event_schedule(year)
            available_races = len(schedule)
            
            # Para años futuros, verificar cuáles han ocurrido
            actual_races = self._count_completed_races(year, available_races, max_races)
            races_to_get = min(actual_races, max_races)


            return self._build_race_list(year, races_to_get, schedule)
            
        except Exception as e:
            print(f"Error obteniendo calendario de {year}: {e}")
            return self._fallback_races(year)
    
    def _count_completed_races(self, year, available_races, max_races):
        """Cuenta carreras completadas para años actuales/futuros"""
        if year < 2025:
            return available_races
        
        completed = 0
        
        for round_num in range(1, min(available_races + 1, max_races + 1)):
            try:
                session = fastf1.get_session(year, round_num, 'R')
                if session.date < pd.Timestamp.now():
                    completed += 1
                else:
                    print(f"   Carrera {round_num} aún no ocurrió")
                    break
            except Exception:
                print(f"   No hay datos para carrera {round_num}")
                break
        
        return completed
    
    def _build_race_list(self, year, races_count, schedule):
        """Construye lista de carreras para el año"""
        races = []
        
        for round_num in range(1, races_count + 1):
            try:
                race_info = schedule[schedule['RoundNumber'] == round_num]
                race_name = race_info.iloc[0]['EventName'] if not race_info.empty else f"Race_{round_num}"
                
                races.append({
                    'year': year,
                    'race_name': race_name,
                    'round_number': round_num
                })
            except Exception as e:
                print(f"   Error con carrera {round_num}: {e}")
                races.append({
                    'year': year,
                    'race_name': f"Race_{round_num}",
                    'round_number': round_num
                })
        
        return races
    
    def _fallback_races(self, year):
        """Fallback para cuando falla la obtención del calendario"""
        fallback_count = 24 if year <= 2024 else 13
        print(f"   Usando fallback: {fallback_count} carreras")
        
        return [{
            'year': year,
            'race_name': f"Race_{i}",
            'round_number': i
        } for i in range(1, fallback_count + 1)]