"""
backend/schemas/user_profile.py

UserProfile — the personalization context shared by every AI service in
backend/ai/. Kept independent of any single engine so it can be reused
by the Disease Engine, Recommendation Engine, and the AI layer alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """A user's health profile used to personalize AI responses."""

    user_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None  # "male" | "female" | "other" | None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    health_conditions: List[str] = field(default_factory=list)
    # e.g. ["Diabetes", "Hypertension"] — should align with DiseaseEngine condition names
    fitness_goal: Optional[str] = None  # "Weight Loss" | "Muscle Gain" | "Maintenance" | None
    allergies: List[str] = field(default_factory=list)
    dietary_preferences: List[str] = field(default_factory=list)
    # e.g. ["vegetarian", "low-sodium", "no artificial sweeteners"]
    daily_calorie_target: Optional[int] = None
    activity_level: Optional[str] = None  # "sedentary" | "moderate" | "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "age": self.age,
            "sex": self.sex,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "health_conditions": self.health_conditions,
            "fitness_goal": self.fitness_goal,
            "allergies": self.allergies,
            "dietary_preferences": self.dietary_preferences,
            "daily_calorie_target": self.daily_calorie_target,
            "activity_level": self.activity_level,
        }

    def to_prompt_summary(self) -> str:
        """Compact, natural-language description of the profile for LLM prompts."""
        parts: List[str] = []
        if self.age:
            parts.append(f"{self.age} years old")
        if self.sex:
            parts.append(self.sex)
        if self.health_conditions:
            parts.append(f"managing: {', '.join(self.health_conditions)}")
        if self.fitness_goal:
            parts.append(f"goal: {self.fitness_goal}")
        if self.allergies:
            parts.append(f"allergic to: {', '.join(self.allergies)}")
        if self.dietary_preferences:
            parts.append(f"dietary preferences: {', '.join(self.dietary_preferences)}")
        if not parts:
            return "No specific health profile provided; give general guidance."
        return "User is " + "; ".join(parts) + "."

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data.get("user_id"),
            age=data.get("age"),
            sex=data.get("sex"),
            height_cm=data.get("height_cm"),
            weight_kg=data.get("weight_kg"),
            health_conditions=data.get("health_conditions", []) or [],
            fitness_goal=data.get("fitness_goal"),
            allergies=data.get("allergies", []) or [],
            dietary_preferences=data.get("dietary_preferences", []) or [],
            daily_calorie_target=data.get("daily_calorie_target"),
            activity_level=data.get("activity_level"),
        )
