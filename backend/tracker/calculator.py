
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


def format_logged_item(log: DailyLog) -> dict:
    """Formats a single DailyLog (with its related FoodItem) for frontend rendering."""
    food_item = log.food_item
    return {
        "log_id": log.id,
        "food_item_id": log.food_item_id,
        "name": food_item.name if food_item else "Unknown item",
        "servings_consumed": log.servings,
        "calories": log.calories,
        "protein": log.protein,
        "carbs": log.carbs,
        "fats": log.fats,
        "ingredients": (food_item.ingredients or []) if food_item else [],
        "allergens": (food_item.allergens or []) if food_item else [],
        "logged_at": log.logged_at.isoformat() if log.logged_at else None,
    }


def format_logged_items(logs: list[DailyLog]) -> list[dict]:
    """Formats a list of DailyLog rows into clean dicts for the frontend, newest first."""
    sorted_logs = sorted(logs, key=lambda l: l.logged_at or datetime.min, reverse=True)
    return [format_logged_item(log) for log in sorted_logs]


def get_daily_summary(target_date: date_cls) -> dict:
    """
    Aggregates all DailyLog entries for a specific date into total macros
    plus the formatted list of individual logged meals.

    Args:
        target_date: a datetime.date to aggregate logs for.

    Returns:
        {
            "date": "YYYY-MM-DD",
            "totals": {"calories": float, "protein": float, "carbs": float, "fats": float},
            "meal_count": int,
            "meals": [ {...formatted log...}, ... ]
        }
    """
    logs = (
        DailyLog.query.filter(DailyLog.log_date == target_date)
        .join(FoodItem, DailyLog.food_item_id == FoodItem.id)
        .all()
    )

    totals = {
        "calories": round(sum(log.calories for log in logs), ROUND_DECIMALS),
        "protein": round(sum(log.protein for log in logs), ROUND_DECIMALS),
        "carbs": round(sum(log.carbs for log in logs), ROUND_DECIMALS),
        "fats": round(sum(log.fats for log in logs), ROUND_DECIMALS),
    }

    return {
        "date": target_date.isoformat(),
        "totals": totals,
        "meal_count": len(logs),
        "meals": format_logged_items(logs),
    }
