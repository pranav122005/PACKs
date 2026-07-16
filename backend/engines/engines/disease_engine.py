"""
backend/engines/disease_engine.py

Disease Engine — evaluates a product's suitability for people managing
specific health conditions or fitness goals:

    Diabetes, Hypertension, Heart Disease, Kidney Disease, Pregnancy,
    Children, Weight Loss, Muscle Gain, PCOS, Thyroid.

Each assessment returns Risk / Reason / Recommendation, driven entirely
by thresholds in health_rules.json (no hardcoded numbers here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.schemas.common import DiseaseAssessment, RiskLevel
from backend.utils.config_loader import HealthRulesConfig, get_health_rules_config


@dataclass
class DiseaseReport:
    """Aggregate output of the Disease Engine for a single product."""

    assessments: List[DiseaseAssessment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"assessments": [a.to_dict() for a in self.assessments]}

    def high_risk_conditions(self) -> List[str]:
        return [
            a.condition
            for a in self.assessments
            if a.risk in (RiskLevel.HIGH, RiskLevel.VERY_HIGH)
        ]


class DiseaseEngine:
    """Runs condition-specific rule sets against a product's nutrient profile."""

    def __init__(self, config: Optional[HealthRulesConfig] = None) -> None:
        self._config = config or get_health_rules_config()
        self._thresholds = self._config.section("disease_thresholds")

    def analyze(self, product: Dict[str, Any]) -> DiseaseReport:
        """Run every condition rule set and return a combined DiseaseReport."""
        nutriments = product.get("nutriments", {}) or {}
        ingredients_text = (product.get("ingredients_text") or "").lower()

        facts = self._extract_facts(nutriments)

        report = DiseaseReport()
        report.assessments.append(self._assess_diabetes(facts, ingredients_text))
        report.assessments.append(self._assess_hypertension(facts))
        report.assessments.append(self._assess_heart_disease(facts))
        report.assessments.append(self._assess_kidney_disease(facts, ingredients_text))
        report.assessments.append(self._assess_pregnancy(facts, ingredients_text))
        report.assessments.append(self._assess_children(facts, ingredients_text))
        report.assessments.append(self._assess_weight_loss(facts))
        report.assessments.append(self._assess_muscle_gain(facts))
        report.assessments.append(self._assess_pcos(facts))
        report.assessments.append(self._assess_thyroid(facts, ingredients_text))
        return report

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_facts(self, nutriments: Dict[str, Any]) -> Dict[str, Optional[float]]:
        salt = self._to_float(nutriments.get("salt_100g"))
        sodium = self._to_float(nutriments.get("sodium_100g"))
        if sodium is None and salt is not None:
            sodium = round((salt * 1000) / 2.5, 2)
        if salt is None and sodium is not None:
            salt = round((sodium * 2.5) / 1000, 3)

        return {
            "calories": self._to_float(nutriments.get("energy_kcal_100g")),
            "sugar": self._to_float(nutriments.get("sugars_100g")),
            "salt": salt,
            "sodium": sodium,
            "protein": self._to_float(nutriments.get("proteins_100g")),
            "fiber": self._to_float(nutriments.get("fiber_100g")),
            "fat": self._to_float(nutriments.get("fat_100g")),
            "saturated_fat": self._to_float(nutriments.get("saturated_fat_100g")),
            "trans_fat": self._to_float(nutriments.get("trans_fat_100g")),
        }

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    # ------------------------------------------------------------------
    # Condition assessments
    # ------------------------------------------------------------------

    def _assess_diabetes(self, f: Dict[str, Optional[float]], ingredients_text: str) -> DiseaseAssessment:
        t = self._thresholds["diabetes"]
        sugar = f["sugar"]
        fiber = f["fiber"] or 0.0
        factors: List[str] = []

        has_artificial_sweetener = self._contains_any(
            ingredients_text, ["aspartame", "sucralose", "acesulfame", "saccharin", "cyclamate"]
        )

        if sugar is None:
            risk = RiskLevel.UNKNOWN
            reason = "Sugar content is not available for this product."
            recommendation = "Check the label directly for total sugar content before consuming."
        elif sugar >= t["sugar_high_g"]:
            risk = RiskLevel.HIGH
            factors.append(f"High sugar content ({sugar}g/100g)")
            reason = (
                f"This product contains {sugar}g of sugar per 100g, which can cause a sharp "
                "rise in blood glucose levels."
            )
            recommendation = "Avoid or consume in very small portions; monitor blood glucose closely if consumed."
        elif sugar >= t["sugar_moderate_g"]:
            risk = RiskLevel.MODERATE
            factors.append(f"Moderate sugar content ({sugar}g/100g)")
            reason = f"This product contains a moderate amount of sugar ({sugar}g/100g)."
            recommendation = "Consume in moderation and pair with fiber or protein to slow glucose absorption."
        else:
            risk = RiskLevel.LOW
            reason = f"Sugar content is low ({sugar}g/100g), which is generally favorable for blood sugar control."
            recommendation = "Generally suitable, but still monitor total daily carbohydrate intake."

        if fiber >= t["fiber_protective_g"] and risk in (RiskLevel.HIGH, RiskLevel.MODERATE):
            factors.append(f"Protective fiber content ({fiber}g/100g) partially offsets sugar impact")
            reason += f" However, its fiber content ({fiber}g/100g) may help moderate the blood sugar response."

        if has_artificial_sweetener and t.get("artificial_sweetener_caution"):
            factors.append("Contains artificial sweeteners")
            reason += " Contains artificial sweeteners — generally considered a lower-glycemic option but should still be used in moderation."

        return DiseaseAssessment("Diabetes", risk, reason, recommendation, factors)

    def _assess_hypertension(self, f: Dict[str, Optional[float]]) -> DiseaseAssessment:
        t = self._thresholds["hypertension"]
        salt = f["salt"]
        sodium = f["sodium"]
        factors: List[str] = []

        if salt is None and sodium is None:
            risk = RiskLevel.UNKNOWN
            reason = "Salt/sodium content is not available for this product."
            recommendation = "Check the label directly for sodium content before consuming."
        elif (salt is not None and salt >= t["salt_high_g"]) or (
            sodium is not None and sodium >= t["sodium_high_mg"]
        ):
            risk = RiskLevel.HIGH
            factors.append(f"High salt/sodium content ({salt}g salt / {sodium}mg sodium per 100g)")
            reason = "This product is high in salt/sodium, which can raise blood pressure."
            recommendation = "Avoid or consume rarely; individuals with hypertension should limit sodium intake strictly."
        elif (salt is not None and salt >= t["salt_moderate_g"]) or (
            sodium is not None and sodium >= t["sodium_moderate_mg"]
        ):
            risk = RiskLevel.MODERATE
            factors.append(f"Moderate salt/sodium content ({salt}g salt / {sodium}mg sodium per 100g)")
            reason = "This product has a moderate salt/sodium content."
            recommendation = "Consume in limited portions and monitor total daily sodium intake."
        else:
            risk = RiskLevel.LOW
            reason = "Salt/sodium content is low, which is favorable for blood pressure management."
            recommendation = "Generally suitable for a low-sodium diet."

        return DiseaseAssessment("Hypertension", risk, reason, recommendation, factors)

    def _assess_heart_disease(self, f: Dict[str, Optional[float]]) -> DiseaseAssessment:
        t = self._thresholds["heart_disease"]
        sat_fat = f["saturated_fat"]
        trans_fat = f["trans_fat"]
        sodium = f["sodium"]
        factors: List[str] = []

        risk = RiskLevel.LOW
        reasons: List[str] = []

        if trans_fat is not None and trans_fat >= t["trans_fat_high_g"]:
            risk = RiskLevel.HIGH
            factors.append(f"Contains trans fat ({trans_fat}g/100g)")
            reasons.append(f"contains {trans_fat}g of trans fat per 100g, which directly raises cardiovascular risk")

        if sat_fat is not None and sat_fat >= t["saturated_fat_high_g"]:
            if risk != RiskLevel.HIGH:
                risk = RiskLevel.MODERATE
            factors.append(f"High saturated fat ({sat_fat}g/100g)")
            reasons.append(f"is high in saturated fat ({sat_fat}g/100g), which can raise LDL cholesterol")

        if sodium is not None and sodium >= t["sodium_high_mg"]:
            if risk == RiskLevel.LOW:
                risk = RiskLevel.MODERATE
            factors.append(f"High sodium ({sodium}mg/100g)")
            reasons.append(f"is high in sodium ({sodium}mg/100g), which can strain cardiovascular health")

        if not reasons:
            reason = "No significant heart-disease risk factors detected (saturated fat, trans fat, sodium are within acceptable ranges)."
            recommendation = "Generally suitable for a heart-healthy diet."
        else:
            reason = "This product " + "; and ".join(reasons) + "."
            recommendation = (
                "Avoid or minimize consumption if managing heart disease; prefer alternatives with less saturated/trans fat and sodium."
                if risk == RiskLevel.HIGH
                else "Consume in moderation as part of a heart-healthy diet."
            )

        return DiseaseAssessment("Heart Disease", risk, reason, recommendation, factors)

    def _assess_kidney_disease(self, f: Dict[str, Optional[float]], ingredients_text: str) -> DiseaseAssessment:
        t = self._thresholds["kidney_disease"]
        sodium = f["sodium"]
        protein = f["protein"]
        factors: List[str] = []

        has_phosphate_additive = self._contains_any(
            ingredients_text, ["phosphate", "phosphoric acid", "sodium phosphate", "potassium phosphate"]
        )

        risk = RiskLevel.LOW
        reasons: List[str] = []

        if sodium is not None and sodium >= t["sodium_high_mg"]:
            risk = RiskLevel.HIGH
            factors.append(f"High sodium ({sodium}mg/100g)")
            reasons.append(f"is high in sodium ({sodium}mg/100g), which burdens kidney function")

        if protein is not None and protein >= t["protein_high_g"]:
            if risk != RiskLevel.HIGH:
                risk = RiskLevel.MODERATE
            factors.append(f"High protein ({protein}g/100g)")
            reasons.append(f"is high in protein ({protein}g/100g), which may need to be limited in later-stage kidney disease")

        if has_phosphate_additive and t.get("phosphate_additive_caution"):
            risk = RiskLevel.HIGH
            factors.append("Contains phosphate additives")
            reasons.append("contains phosphate additives, which are rapidly absorbed and can accelerate kidney damage")

        if not reasons:
            reason = "No significant kidney-disease risk factors detected in sodium, protein, or phosphate additives."
            recommendation = "Generally suitable, but always confirm against your renal diet plan."
        else:
            reason = "This product " + "; and ".join(reasons) + "."
            recommendation = (
                "Avoid this product; consult a renal dietitian before consuming."
                if risk == RiskLevel.HIGH
                else "Consume only occasionally and discuss with your healthcare provider."
            )

        return DiseaseAssessment("Kidney Disease", risk, reason, recommendation, factors)

    def _assess_pregnancy(self, f: Dict[str, Optional[float]], ingredients_text: str) -> DiseaseAssessment:
        t = self._thresholds["pregnancy"]
        factors: List[str] = []
        concerns: List[str] = []

        if t.get("caffeine_caution") and self._contains_any(ingredients_text, ["caffeine", "coffee", "guarana"]):
            factors.append("Contains caffeine")
            concerns.append("contains caffeine, which should be limited during pregnancy (under ~200mg/day total)")

        if t.get("artificial_sweetener_caution") and self._contains_any(
            ingredients_text, ["saccharin", "cyclamate"]
        ):
            factors.append("Contains sweeteners flagged for pregnancy caution")
            concerns.append("contains artificial sweeteners (e.g. saccharin/cyclamate) that some guidelines advise limiting during pregnancy")

        if t.get("unpasteurized_caution") and self._contains_any(
            ingredients_text, ["unpasteurized", "raw milk"]
        ):
            factors.append("Contains unpasteurized ingredients")
            concerns.append("may contain unpasteurized ingredients, which carry a listeria risk during pregnancy")

        if t.get("additive_caution") and self._contains_any(
            ingredients_text, ["sodium nitrite", "sodium nitrate", "bha", "bht"]
        ):
            factors.append("Contains additives flagged for pregnancy caution")
            concerns.append("contains preservatives (nitrites/nitrates or BHA/BHT) that are advised against during pregnancy")

        if concerns:
            risk = RiskLevel.HIGH if len(concerns) >= 2 else RiskLevel.MODERATE
            reason = "This product " + "; and ".join(concerns) + "."
            recommendation = "Consult your obstetrician before consuming; consider an alternative without these ingredients."
        else:
            risk = RiskLevel.LOW
            reason = "No major pregnancy-specific concerns detected (caffeine, certain sweeteners, unpasteurized ingredients, or flagged preservatives)."
            recommendation = "Generally suitable in moderation as part of a balanced prenatal diet."

        return DiseaseAssessment("Pregnancy", risk, reason, recommendation, factors)

    def _assess_children(self, f: Dict[str, Optional[float]], ingredients_text: str) -> DiseaseAssessment:
        t = self._thresholds["children"]
        sugar = f["sugar"]
        factors: List[str] = []
        concerns: List[str] = []

        if sugar is not None and sugar >= t["sugar_high_g"]:
            factors.append(f"High sugar content ({sugar}g/100g)")
            concerns.append(f"is high in sugar ({sugar}g/100g), which contributes to poor dietary habits and dental issues in children")

        if t.get("artificial_colour_caution") and self._contains_any(
            ingredients_text, ["tartrazine", "sunset yellow", "carmoisine", "ponceau", "quinoline yellow"]
        ):
            factors.append("Contains artificial colours linked to hyperactivity")
            concerns.append("contains artificial colours that have been linked to hyperactivity in some children")

        if t.get("artificial_sweetener_caution") and self._contains_any(
            ingredients_text, ["aspartame", "acesulfame", "saccharin"]
        ):
            factors.append("Contains artificial sweeteners")
            concerns.append("contains artificial sweeteners not recommended for young children")

        if t.get("caffeine_caution") and self._contains_any(ingredients_text, ["caffeine", "guarana"]):
            factors.append("Contains caffeine")
            concerns.append("contains caffeine, which is not recommended for children")

        if concerns:
            risk = RiskLevel.HIGH if len(concerns) >= 2 else RiskLevel.MODERATE
            reason = "This product " + "; and ".join(concerns) + "."
            recommendation = "Limit or avoid giving this product to children; choose a naturally sweetened, additive-free alternative."
        else:
            risk = RiskLevel.LOW
            reason = "No major child-specific concerns detected in sugar, artificial colours, sweeteners, or caffeine."
            recommendation = "Generally suitable for children in age-appropriate portions."

        return DiseaseAssessment("Children", risk, reason, recommendation, factors)

    def _assess_weight_loss(self, f: Dict[str, Optional[float]]) -> DiseaseAssessment:
        t = self._thresholds["weight_loss"]
        calories = f["calories"]
        sugar = f["sugar"]
        fat = f["fat"]
        fiber = f["fiber"] or 0.0
        factors: List[str] = []
        concerns: List[str] = []

        if calories is not None and calories >= t["calories_high_kcal"]:
            factors.append(f"High calorie density ({calories}kcal/100g)")
            concerns.append(f"is calorie-dense ({calories}kcal/100g), which can hinder a calorie deficit")

        if sugar is not None and sugar >= t["sugar_high_g"]:
            factors.append(f"High sugar content ({sugar}g/100g)")
            concerns.append(f"is high in sugar ({sugar}g/100g), contributing empty calories")

        if fat is not None and fat >= t["fat_high_g"]:
            factors.append(f"High fat content ({fat}g/100g)")
            concerns.append(f"is high in fat ({fat}g/100g), which is calorie-dense")

        if concerns:
            risk = RiskLevel.HIGH if len(concerns) >= 2 else RiskLevel.MODERATE
            reason = "This product " + "; and ".join(concerns) + "."
            recommendation = "Limit portion size significantly or choose a lower-calorie, higher-fiber alternative."
        else:
            risk = RiskLevel.LOW
            reason = "This product has a favorable calorie, sugar, and fat profile for weight management."
            recommendation = "Fits reasonably within a calorie-controlled diet."

        if fiber >= t["fiber_protective_g"]:
            factors.append(f"Protective fiber content ({fiber}g/100g)")
            reason += f" Its fiber content ({fiber}g/100g) may help promote satiety."

        return DiseaseAssessment("Weight Loss", risk, reason, recommendation, factors)

    def _assess_muscle_gain(self, f: Dict[str, Optional[float]]) -> DiseaseAssessment:
        t = self._thresholds["muscle_gain"]
        protein = f["protein"]
        sugar = f["sugar"]
        factors: List[str] = []

        if protein is None:
            risk = RiskLevel.UNKNOWN
            reason = "Protein content is not available for this product."
            recommendation = "Check the label directly for protein content."
        elif protein >= t["protein_excellent_g"]:
            risk = RiskLevel.LOW
            factors.append(f"Excellent protein content ({protein}g/100g)")
            reason = f"Excellent protein density ({protein}g/100g), well suited to support muscle protein synthesis."
            recommendation = "Great choice to help meet daily protein targets for muscle gain."
        elif protein >= t["protein_low_g"]:
            risk = RiskLevel.MODERATE
            factors.append(f"Moderate protein content ({protein}g/100g)")
            reason = f"Moderate protein density ({protein}g/100g)."
            recommendation = "Can contribute to daily protein intake but pair with other protein sources."
        else:
            risk = RiskLevel.HIGH
            factors.append(f"Low protein content ({protein}g/100g)")
            reason = f"Low protein density ({protein}g/100g), contributes little toward muscle-building goals."
            recommendation = "Not an efficient protein source; pair with a dedicated protein-rich food."

        if sugar is not None and sugar >= t["sugar_caution_g"]:
            factors.append(f"High sugar content ({sugar}g/100g) alongside protein")
            reason += f" Note it also contains {sugar}g of sugar per 100g."

        return DiseaseAssessment("Muscle Gain", risk, reason, recommendation, factors)

    def _assess_pcos(self, f: Dict[str, Optional[float]]) -> DiseaseAssessment:
        t = self._thresholds["pcos"]
        sugar = f["sugar"]
        trans_fat = f["trans_fat"]
        factors: List[str] = []
        concerns: List[str] = []

        if sugar is not None and sugar >= t["sugar_high_g"]:
            factors.append(f"High sugar content ({sugar}g/100g)")
            concerns.append(f"is high in sugar ({sugar}g/100g), which can worsen insulin resistance associated with PCOS")

        if trans_fat is not None and trans_fat >= t["trans_fat_high_g"]:
            factors.append(f"Contains trans fat ({trans_fat}g/100g)")
            concerns.append("contains trans fat, which is linked to worsened inflammation and insulin resistance")

        if concerns:
            risk = RiskLevel.HIGH if len(concerns) >= 2 else RiskLevel.MODERATE
            reason = "This product " + "; and ".join(concerns) + "."
            recommendation = "Limit or avoid; prefer low-glycemic, low-trans-fat alternatives to help manage insulin resistance."
        else:
            risk = RiskLevel.LOW
            reason = "No major PCOS-specific concerns detected in sugar or trans fat content."
            recommendation = "Generally suitable as part of a low-glycemic, PCOS-friendly diet."

        return DiseaseAssessment("PCOS", risk, reason, recommendation, factors)

    def _assess_thyroid(self, f: Dict[str, Optional[float]], ingredients_text: str) -> DiseaseAssessment:
        t = self._thresholds["thyroid"]
        factors: List[str] = []
        concerns: List[str] = []

        if t.get("soy_caution") and self._contains_any(ingredients_text, ["soy", "soya", "soy protein", "soy lecithin"]):
            factors.append("Contains soy")
            concerns.append("contains soy, which may interfere with thyroid hormone medication absorption if consumed close to dosing")

        if t.get("processed_food_caution") and self._contains_any(
            ingredients_text, ["hydrogenated", "artificial flavour", "artificial flavor"]
        ):
            factors.append("Highly processed formulation")
            concerns.append("is a highly processed formulation, which is generally advised against for thyroid health")

        if concerns:
            risk = RiskLevel.MODERATE
            reason = "This product " + "; and ".join(concerns) + "."
            recommendation = "If on thyroid medication, avoid consuming within 3-4 hours of your dose; otherwise consume in moderation."
        else:
            risk = RiskLevel.LOW
            reason = "No major thyroid-specific concerns detected in soy content or processing level."
            recommendation = "Generally suitable for individuals managing thyroid conditions."

        return DiseaseAssessment("Thyroid", risk, reason, recommendation, factors)
