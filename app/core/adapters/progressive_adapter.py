from app.config import DRIVERS_2025, ADAPTATION_SYSTEM, PREDICTION_CONFIG

class ProgressiveAdapter:
    def __init__(self):
        self.adaptation_config = ADAPTATION_SYSTEM
        self.drivers_config = DRIVERS_2025
        # Fracción mínima de penalización aun estando "completamente adaptado"
        self.min_penalty_fraction = 1.0 / 16.0
        
    def calculate_adaptation_penalty(self, driver, current_race_number):
        """Calcula la penalización progresiva SOLO para cambio de equipo"""
        driver_info = self.drivers_config.get(driver, {})
        # Rookies no usan penalización progresiva
        if not driver_info.get("team_change", False):
            return 0.0
        change_type = "team_change"
        change_config = self.adaptation_config["change_types"][change_type]
        base_penalty = change_config["base_penalty"]
        adaptation_races = change_config["adaptation_races"]
        # Fracción mínima de penalización aun estando "fully_adapted"
        min_frac = getattr(self, "min_penalty_fraction", 0.125)
        # Asegurar límites válidos
        if adaptation_races <= 0:
            return base_penalty * min_frac
        # Calcular fracción restante de penalización de forma lineal, con piso min_frac
        if current_race_number <= 1:
            frac = 1.0  # 100% cuando aún no corrió con el nuevo team
        elif current_race_number <= adaptation_races:
            progress_ratio = (current_race_number - 1) / adaptation_races
            frac = max(min_frac, 1.0 - progress_ratio)
        else:
            # Completamente adaptado: se mantiene la penalización mínima
            frac = min_frac
        remaining_penalty = base_penalty * max(0.0, min(1.0, frac))
        return remaining_penalty
    
    def get_adaptation_status(self, driver, current_race_number):
        """Estado de adaptación SOLO para cambio de equipo; rookies se tratan aparte"""
        driver_info = self.drivers_config.get(driver, {})
        if not driver_info.get("team_change", False):
            return {
                "status": "fully_adapted",
                "penalty": 0.0,
                "description": "Sin cambio de equipo",
                "progress": 100
            }
        change_type = "team_change"
        change_config = self.adaptation_config["change_types"][change_type]
        adaptation_races = change_config["adaptation_races"]
        penalty = self.calculate_adaptation_penalty(driver, current_race_number)
        if current_race_number > adaptation_races:
            status = "fully_adapted"
            progress = 100
        else:
            status = "adapting"
            progress = min(100, int((current_race_number / adaptation_races) * 100))

        return {
            "status": status,
            "penalty": penalty,
            "description": change_config["description"],
            "progress": progress,
            "races_remaining": max(0, adaptation_races - current_race_number + 1)
        }
    
    def apply_progressive_penalties(self, predictions_df, current_race_number):
        """Aplica penalizaciones progresivas a las predicciones"""
        print(f"📊 Aplicando adaptación progresiva para la carrera #{current_race_number}")
        
        adaptations_applied = []
        
        for idx, row in predictions_df.iterrows():
            driver = row['driver']
            adaptation_status = self.get_adaptation_status(driver, current_race_number)
            penalty = adaptation_status["penalty"]
            
            if penalty > 0:
                # Aplicar penalización a la posición predicha
                old_position = row['predicted_position']
                new_position = old_position + penalty
                predictions_df.loc[idx, 'predicted_position'] = new_position
                
                # Agregar información de adaptación
                predictions_df.loc[idx, 'adaptation_penalty'] = penalty
                predictions_df.loc[idx, 'adaptation_progress'] = adaptation_status["progress"]
                
                adaptations_applied.append({
                    'driver': driver,
                    'old_position': old_position,
                    'new_position': new_position,
                    'penalty': penalty,
                    'status': adaptation_status["status"],
                    'progress': adaptation_status["progress"],
                    'description': adaptation_status["description"]
                })
        
        # Mostrar resumen de adaptaciones
        self._show_adaptation_summary(adaptations_applied, current_race_number)
        
        return predictions_df
    
    def _show_adaptation_summary(self, adaptations, current_race_number):
        """Muestra un resumen de las adaptaciones aplicadas"""
        if not adaptations:
            print("   ✅ Todos los pilotos ya están completamente adaptados")
            return
        

        base_penalty = self.adaptation_config['change_types']['team_change']['base_penalty']
        for adaptation in adaptations:
            frac_pct = (adaptation['penalty'] / base_penalty * 100.0) if base_penalty else 0.0
            print(
                f"{adaptation['driver']:<6} "
                f"P{adaptation['old_position']:<6.2f} "
                f"P{adaptation['new_position']:<8.2f} "
                f"{adaptation['penalty']:<8.2f} "
                f"{frac_pct:<7.1f}% "
                f"{adaptation['progress']:<8d}% "
                f"{adaptation['description']}"
            )