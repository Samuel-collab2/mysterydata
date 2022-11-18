from dataclasses import dataclass, field

from core.model_induction import train_random_forest
from core.model_regression import train_linear_regression
from core.model_set_modifiers import modifier_filter_expanded


@dataclass(frozen=True)
class ModelSet:
    name: str
    train_induction_model: callable = field(default=train_random_forest)
    induction_modifiers: list = field(default_factory=lambda: [modifier_filter_expanded()])
    train_regression_model: callable = field(default=train_linear_regression)
    regression_modifiers: list = field(default_factory=lambda: [modifier_filter_expanded()])
    proba_threshold: bool = 0.5
