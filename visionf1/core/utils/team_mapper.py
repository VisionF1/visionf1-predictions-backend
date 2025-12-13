#!/usr/bin/env python3
"""
Team Mapper - Mapeo de equipos históricos a sus nombres actuales en 2025
Mantiene la trazabilidad histórica para predicciones más precisas
"""

class F1TeamMapper:
    """
    Clase para mapear equipos históricos a sus equivalentes actuales en 2025.
    Permite mantener la continuidad histórica para análisis y predicciones.
    """
    
    def __init__(self):
        """
        Inicializa el mapeador con las transformaciones históricas de equipos F1.
        Basado en el análisis real del dataset 2022-2025.
        """
        
        # Mapeo de equipos históricos a sus equivalentes en 2025
        self.team_mapping = {
            # AlphaTauri -> RB -> Racing Bulls (Evolución de Red Bull's sister team)
            'AlphaTauri': 'Racing Bulls',
            'RB': 'Racing Bulls',
            
            # Alfa Romeo -> Kick Sauber (Cambio de marca/naming sponsor)
            'Alfa Romeo': 'Kick Sauber',
            
            # Equipos que se mantuvieron consistentes 2022-2025
            'Alpine': 'Alpine',
            'Aston Martin': 'Aston Martin', 
            'Ferrari': 'Ferrari',
            'Haas F1 Team': 'Haas F1 Team',
            'McLaren': 'McLaren',
            'Mercedes': 'Mercedes',
            'Red Bull Racing': 'Red Bull Racing',
            'Williams': 'Williams'
        }
        
        # Mapeo inverso para obtener todos los nombres históricos de un equipo actual
        self.reverse_mapping = self._create_reverse_mapping()
        
        # Información adicional sobre las transformaciones
        self.team_evolution = {
            'Racing Bulls': {
                'previous_names': ['AlphaTauri', 'RB'],
                'years_active': {
                    'AlphaTauri': [2022, 2023],
                    'RB': [2024],
                    'Racing Bulls': [2025]
                },
                'parent_company': 'Red Bull',
                'transformation_type': 'rebranding',
                'notes': 'Sister team de Red Bull Racing, cambios de marca/sponsor'
            },
            'Kick Sauber': {
                'previous_names': ['Alfa Romeo'],
                'years_active': {
                    'Alfa Romeo': [2022, 2023],
                    'Kick Sauber': [2024, 2025]
                },
                'parent_company': 'Sauber Motorsport',
                'transformation_type': 'sponsor_change',
                'notes': 'Cambio de naming sponsor, misma estructura de equipo'
            }
        }
        
        # Equipos estables que no han cambiado
        self.stable_teams = [
            'Alpine', 'Aston Martin', 'Ferrari', 'Haas F1 Team',
            'McLaren', 'Mercedes', 'Red Bull Racing', 'Williams'
        ]
    
    def _create_reverse_mapping(self):
        """Crea el mapeo inverso: equipo_2025 -> [nombres_historicos]"""
        reverse_map = {}
        for historical_name, current_name in self.team_mapping.items():
            if current_name not in reverse_map:
                reverse_map[current_name] = []
            if historical_name != current_name:  # Solo agregar si es diferente
                reverse_map[current_name].append(historical_name)
        return reverse_map
    
    def map_to_current(self, historical_team_name):
        """
        Mapea un nombre histórico de equipo a su equivalente actual en 2025.
        
        Args:
            historical_team_name (str): Nombre del equipo en años anteriores
            
        Returns:
            str: Nombre del equipo equivalente en 2025
        """
        return self.team_mapping.get(historical_team_name, historical_team_name)
    
    def get_historical_names(self, current_team_name):
        """
        Obtiene todos los nombres históricos de un equipo actual.
        
        Args:
            current_team_name (str): Nombre del equipo en 2025
            
        Returns:
            list: Lista de nombres históricos del equipo
        """
        historical_names = self.reverse_mapping.get(current_team_name, [])
        # Incluir el nombre actual también
        if current_team_name not in historical_names:
            historical_names.append(current_team_name)
        return historical_names
    
    def get_team_evolution_info(self, team_name):
        """
        Obtiene información detallada sobre la evolución de un equipo.
        
        Args:
            team_name (str): Nombre del equipo (histórico o actual)
            
        Returns:
            dict: Información sobre la evolución del equipo
        """
        # Mapear a nombre actual primero
        current_name = self.map_to_current(team_name)
        return self.team_evolution.get(current_name, {
            'previous_names': [],
            'years_active': {current_name: [2022, 2023, 2024, 2025]},
            'transformation_type': 'stable',
            'notes': 'Equipo estable sin cambios de nombre'
        })
    
    def is_same_team_lineage(self, team1, team2):
        """
        Determina si dos nombres de equipos pertenecen al mismo linaje/organización.
        
        Args:
            team1 (str): Primer nombre de equipo
            team2 (str): Segundo nombre de equipo
            
        Returns:
            bool: True si pertenecen al mismo linaje
        """
        current1 = self.map_to_current(team1)
        current2 = self.map_to_current(team2)
        return current1 == current2
    
    def get_all_teams_2025(self):
        """
        Obtiene lista de todos los equipos que existían en 2025.
        
        Returns:
            list: Nombres de equipos en 2025
        """
        return list(set(self.team_mapping.values()))
    
    def get_transformation_summary(self):
        """
        Genera un resumen de todas las transformaciones de equipos.
        
        Returns:
            dict: Resumen de transformaciones
        """
        transformations = {}
        stable_teams = []
        
        for current_team in self.get_all_teams_2025():
            historical_names = self.get_historical_names(current_team)
            if len(historical_names) > 1:  # Hubo transformaciones
                transformations[current_team] = {
                    'historical_names': [name for name in historical_names if name != current_team],
                    'evolution_info': self.get_team_evolution_info(current_team)
                }
            else:
                stable_teams.append(current_team)
        
        return {
            'transformations': transformations,
            'stable_teams': stable_teams,
            'total_teams_2025': len(self.get_all_teams_2025()),
            'teams_with_changes': len(transformations),
            'stable_teams_count': len(stable_teams)
        }
    
    def apply_mapping_to_dataframe(self, df, team_column='team'):
        """
        Aplica el mapeo de equipos a un DataFrame, actualizando los nombres históricos.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos de F1
            team_column (str): Nombre de la columna que contiene los equipos
            
        Returns:
            pandas.DataFrame: DataFrame con equipos mapeados a nombres 2025
        """
        df_mapped = df.copy()
        df_mapped[f'{team_column}_historical'] = df_mapped[team_column].copy()  # Preservar nombres originales
        df_mapped[team_column] = df_mapped[team_column].map(self.map_to_current)
        return df_mapped
    
    def print_mapping_summary(self):
        """Imprime un resumen legible del mapeo de equipos."""
        print("🏁 F1 TEAM MAPPER - RESUMEN DE TRANSFORMACIONES")
        print("=" * 60)
        
        summary = self.get_transformation_summary()
        
        print(f"📊 Total equipos en 2025: {summary['total_teams_2025']}")
        print(f"🔄 Equipos con cambios: {summary['teams_with_changes']}")
        print(f"✅ Equipos estables: {summary['stable_teams_count']}")
        
        print("\n🔄 TRANSFORMACIONES HISTÓRICAS:")
        for current_team, info in summary['transformations'].items():
            print(f"\n🏎️  {current_team}:")
            evolution = info['evolution_info']
            
            if 'years_active' in evolution:
                for historical_name, years in evolution['years_active'].items():
                    years_str = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])
                    if historical_name == current_team:
                        print(f"   ✅ {historical_name}: {years_str}")
                    else:
                        print(f"   📅 {historical_name}: {years_str}")
            
            if 'transformation_type' in evolution:
                print(f"   🔧 Tipo: {evolution['transformation_type']}")
            
            if 'notes' in evolution:
                print(f"   📝 Notas: {evolution['notes']}")
        
        print("\n✅ EQUIPOS ESTABLES (sin cambios):")
        for team in summary['stable_teams']:
            print(f"   🏁 {team}")


def main():
    """Función principal para testing del mapper."""
    mapper = F1TeamMapper()
    
    # Mostrar resumen
    mapper.print_mapping_summary()
    
    # Ejemplos de uso
    print("\n🧪 EJEMPLOS DE USO:")
    print("-" * 30)
    
    # Mapeo de histórico a actual
    print(f"AlphaTauri (2022) -> {mapper.map_to_current('AlphaTauri')}")
    print(f"RB (2024) -> {mapper.map_to_current('RB')}")
    print(f"Alfa Romeo (2022) -> {mapper.map_to_current('Alfa Romeo')}")
    
    # Nombres históricos de un equipo actual
    print(f"\nRacing Bulls históricos: {mapper.get_historical_names('Racing Bulls')}")
    print(f"Kick Sauber históricos: {mapper.get_historical_names('Kick Sauber')}")
    
    # Verificar si son el mismo linaje
    print(f"\n¿AlphaTauri y Racing Bulls mismo linaje? {mapper.is_same_team_lineage('AlphaTauri', 'Racing Bulls')}")
    print(f"¿Mercedes y Ferrari mismo linaje? {mapper.is_same_team_lineage('Mercedes', 'Ferrari')}")


if __name__ == "__main__":
    main()
