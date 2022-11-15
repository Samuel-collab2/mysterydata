from shapely.geometry import Point, Polygon

from core.model_induction_wrapper import ModelInductionWrapper


def get_point(claim, x_axis, y_axis):
    return Point(claim[x_axis], claim[y_axis])


def in_boundary(claim, x_axis, y_axis, coordinates):
    return Polygon(coordinates).contains(get_point(claim, x_axis, y_axis))


def evaluate_conditions(conditions):
    for condition in conditions:
        if condition:
            return True
    return False

def _should_accept_claim(claim):
    return False


def _should_reject_claim(claim):
    return evaluate_conditions([
        in_boundary(claim, "feature1", "feature2", [
            (-1, -1),
            (-1, 1000),
            (0.07, 1000),
            (0.07, -1),
        ]),
        in_boundary(claim, "feature1", "feature6", [
            (-1, -1),
            (-1, 30.5),
            (0.1, 30.5),
            (0.1, -1),
        ]),
        in_boundary(claim, "feature1", "feature6", [
            (-1, 33),
            (-1, 55),
            (0.25, 55),
            (0.05, 33),
        ]),
        in_boundary(claim, "feature1", "feature6", [
            (-1, 0.05),
            (-1, 9.95),
            (2, 9.95),
            (2, 0.05),
        ]),
        in_boundary(claim, "feature1", "feature6", [
            (-1, 10.05),
            (-1, 24.95),
            (2, 24.95),
            (2, 10.05),
        ]),
        in_boundary(claim, "feature1", "feature8", [
            (-1, -1),
            (-1, 200),
            (0.07, 200),
            (0.07, -1),
        ]),
        in_boundary(claim, "feature1", "feature10", [
            (-1, -1),
            (-1, 300),
            (0.07, 300),
            (0.07, -1),
        ]),
        in_boundary(claim, "feature1", "feature12", [
            (-1, -1),
            (-1, 30),
            (0.07, 30),
            (0.07, -1),
        ]),
        in_boundary(claim, "feature1", "feature12", [
            (-1, -1),
            (-1, 2.5),
            (2, 2.5),
            (2, -1),
        ]),
        in_boundary(claim, "feature1", "feature12", [
            (-1, 15.5),
            (-1, 30),
            (2, 30),
            (2, 15.5),
        ]),
    ])

def train_boundary(train_features, train_label, model):
    model_wrapper = ModelInductionWrapper(
        model,
        predicate_accept=_should_accept_claim,
        predicate_reject=_should_reject_claim
    )

    model_wrapper.fit(train_features, train_label)

    return model_wrapper
