"""
Specialized Sports Science & Calculation Tools for the AI Personal Trainer Agent.
"""

class PersonalTrainerTools:
    
    @staticmethod
    def calculate_one_rep_max(weight_kg: float, reps: int) -> float:
        """
        Estimates One-Rep Maximum (1RM) using Brzycki's formula:
        1RM = weight / (1.0278 - 0.0278 * reps)
        """
        if reps <= 0:
            return 0.0
        if reps == 1:
            return weight_kg
        if reps > 36:
            # Formula loses accuracy past ~36 reps
            reps = 36
            
        estimated_1rm = weight_kg / (1.0278 - (0.0278 * reps))
        return round(estimated_1rm, 2)

    @staticmethod
    def calculate_target_load_for_rir(estimated_1rm: float, target_rir: int) -> dict:
        """
        Estimates working weight percentages based on target RIR / RPE intensity zones
        derived from NSCA velocity/percentage tables.
        Approximate %1RM mapping:
        - RIR 0 (Failure): ~100% of 1RM for 1 rep, ~85% for 8 reps
        - RIR 2: ~80-82% for 8 reps
        """
        # General percentage estimates relative to 1RM for standard hypertrophy rep ranges (8-12 reps)
        intensity_map = {
            0: 0.82,  # RIR 0 (~82% for hypertrophy ranges)
            1: 0.79,  # RIR 1 (~79%)
            2: 0.76,  # RIR 2 (~76%)
            3: 0.73   # RIR 3 (~73%)
        }
        
        factor = intensity_map.get(target_rir, 0.75)
        recommended_weight = estimated_1rm * factor
        
        return {
            "target_rir": target_rir,
            "estimated_1rm": estimated_1rm,
            "recommended_working_weight": round(recommended_weight, 1),
            "intensity_percentage": f"{int(factor * 100)}%"
        }