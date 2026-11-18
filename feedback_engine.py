from typing import Dict, Any


class TrainingFeedbackEngine:
    def __init__(self):
        # Thresholds for RPE and RIR decision-making
        self.rpe_high_threshold = 9.5
        self.rpe_low_threshold = 7.0

    def analyze_feedback(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the student's post-workout feedback.
        
        Expected session_data:
        {
            "exercise": "Barbell Bench Press",
            "target_rir": 2,
            "actual_rir": 0,
            "rpe": 10.0,
            "fatigue_level": "high", # low, moderate, high, extreme
            "completed_all_sets": True,
            "joint_discomfort": False
        }
        """
        rpe = session_data.get("rpe", 8.0)
        actual_rir = session_data.get("actual_rir", 2)
        target_rir = session_data.get("target_rir", 2)
        fatigue = session_data.get("fatigue_level", "moderate")
        discomfort = session_data.get("joint_discomfort", False)

        adjustment = {
            "action": "maintain",
            "load_modifier": 0.0,  # load adjustment percentage (e.g., +0.02 for +2%)
            "volume_modifier": 0,  # adjustment in number of working sets
            "reason": ""
        }

        # Rule 1: Joint discomfort or acute pain -> Immediate exercise substitution or deload
        if discomfort:
            adjustment["action"] = "substitute_exercise"
            adjustment["reason"] = "Joint discomfort reported. Exercise substitution with lower shear stress/joint strain is recommended."
            return adjustment

        # Rule 2: Extreme fatigue or consistent RPE 10 below target RIR -> Risk of excessive overreaching
        if fatigue == "extreme" or (rpe >= 9.5 and actual_rir < target_rir - 1):
            adjustment["action"] = "deload_or_drop"
            adjustment["load_modifier"] = -0.05  # Reduce load by 5%
            adjustment["volume_modifier"] = -1   # Remove 1 working set
            adjustment["reason"] = "Signs of accumulated systemic fatigue or performance drop. Reducing volume and load for recovery."
            return adjustment

        # Rule 3: Progressive overload if RIR was higher than target (too easy)
        if actual_rir > target_rir + 1 and fatigue in ["low", "moderate"]:
            adjustment["action"] = "progress_load"
            adjustment["load_modifier"] = 0.03  # Increase load by 3% to 5% in the next session
            adjustment["reason"] = "Performance above expectations with comfortable RIR reserve. Load progression indicated."
            return adjustment

        # Default: Maintain current progression
        adjustment["action"] = "maintain"
        adjustment["reason"] = "Stable progress within the target exertion zone (RIR/RPE)."
        return adjustment


if __name__ == "__main__":
    engine = TrainingFeedbackEngine()
    
    # Test sample with an exhausted student
    sample_feedback = {
        "exercise": "Barbell Squat",
        "target_rir": 2,
        "actual_rir": 0,
        "rpe": 10.0,
        "fatigue_level": "extreme",
        "completed_all_sets": True,
        "joint_discomfort": False
    }
    
    result = engine.analyze_feedback(sample_feedback)
    print("Feedback Analysis Result:", result)