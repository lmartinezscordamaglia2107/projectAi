"""
Schema package exposing validated Pydantic models for Athlete Profiles,
Workout Prescriptions, and Post-Workout Autoregulation Telemetry,
along with robust JSON parsing and serialization utilities.
"""

import json
from typing import Any, Optional, Union, Type
from pydantic import BaseModel

# Importing core validated schemas
from src.schema.athlete import AthleteProfile, BiomechanicalConstraint
from src.schema.feedback import SessionFeedbackInput, FeedbackAdjustmentOutput
from src.schema.workout import ExercisePrescription, WorkoutSession, WeeklyTrainingPlan

__all__ = [
    "AthleteProfile",
    "BiomechanicalConstraint",
    "SessionFeedbackInput",
    "FeedbackAdjustmentOutput",
    "ExercisePrescription",
    "WorkoutSession",
    "WeeklyTrainingPlan",
    "parse_json_to_model",
    "model_to_json_str",
]


def parse_json_to_model(model_class: Type[BaseModel], json_input: Union[str, dict]) -> Optional[BaseModel]:
    """
    Safely parses a JSON string or dictionary into any validated Pydantic schema model.
    
    Args:
        model_class: The target Pydantic class (e.g., AthleteProfile, SessionFeedbackInput).
        json_input: Raw JSON string or Python dictionary containing the payload.
    
    Returns:
        An instance of the model_class, or None if parsing fails.
    """
    try:
        if isinstance(json_input, str):
            data_dict = json.loads(json_input)
        elif isinstance(json_input, dict):
            data_dict = json_input
        else:
            return None
        
        return model_class(**data_dict)
    except Exception as e:
        class_name = getattr(model_class, "__name__", "UnknownModel")
        print(f"[Schema Parser Error] Failed to parse JSON into {class_name}: {e}")
        return None


def model_to_json_str(model_instance: Any, indent: int = 4) -> Optional[str]:
    """
    Serializes any Pydantic model instance into a formatted JSON string using Pydantic v2 standards.
    """
    try:
        if hasattr(model_instance, "model_dump_json"):
            return model_instance.model_dump_json(indent=indent)
        elif hasattr(model_instance, "model_dump"):
            return json.dumps(model_instance.model_dump(), indent=indent, ensure_ascii=False)
        elif hasattr(model_instance, "dict"):
            # Fallback for older Pydantic v1 instances if any exist
            return json.dumps(model_instance.dict(), indent=indent, ensure_ascii=False)
        return None
    except Exception as e:
        print(f"[Schema Serializer Error] Failed to serialize model: {e}")
        return None