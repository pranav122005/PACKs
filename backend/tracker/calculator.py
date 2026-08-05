
import logging
from datetime import date as date_cls, datetime

from backend.database.models import DailyLog, FoodItem

logger = logging.getLogger(__name__)

ROUND_DECIMALS = 2


def calculate_scaled_macros(food_item: FoodItem, servings: float) -> dict:
    """
    Calculates total macros for a FoodItem scaled by the number of servings
    actually consumed.

    Args:
        food_item: a FoodItem instance holding per-serving base macros.
        servings: number of servings consumed (e.g. 1.5).

    Returns:
        dict with keys: calories, protein, carbs, fats (all floats, rounded).

    Raises:
        ValueError: if servings is not a positive number.
    """
    if servings is None or servings <= 0:
        raise ValueError("servings must be a positive number.")

    return {
        "calories": round(food_item.calories * servings, ROUND_DECIMALS),
        "protein": round(food_item.protein * servings, ROUND_DECIMALS),
        "carbs": round(food_item.carbs * servings, ROUND_DECIMALS),
        "fats": round(food_item.fats * servings, ROUND_DECIMALS),
    }


def parse_date_string(date_str: str | None) -> date_cls:
    """
    Parses a 'YYYY-MM-DD' date string. Defaults to today if None/empty.

    Raises:
        ValueError: if date_str is present but not valid YYYY-MM-DD.
    """
    if not date_str:
        return date_cls.today()

    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.") from exc


