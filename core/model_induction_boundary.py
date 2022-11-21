from shapely.geometry import Point, Polygon

from core.model_induction_wrapper import ModelInductionWrapper


def get_point(claim, x_axis, y_axis):
    return Point(claim[x_axis], claim[y_axis])


def in_boundary(claim, x_axis, y_axis, coordinates):
    return Polygon(coordinates).contains(get_point(claim, x_axis, y_axis))

def in_rectangle(claim, x_axis, y_axis, x_min, x_max, y_min, y_max):
    return in_boundary(claim, x_axis, y_axis, [
        (x_min, y_min),
        (x_min, y_max),
        (x_max, y_max),
        (x_max, y_min),
    ]),

def evaluate_conditions(conditions):
    for condition in conditions:
        if condition:
            return True
    return False

def _should_accept_claim(claim):
    return False


def _should_reject_claim_1(claim):
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

def _should_reject_claim_2(claim):
    return evaluate_conditions([
        in_boundary(claim, "feature1", "feature2", [
            (-1, -1),
            (-1, 1000),
            (0.07, 1000),
            (0.07, -1),
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
        in_boundary(claim, "feature1", "feature17", [
            (-0.1, 12.5),
            (0.8, 0),
            (0, 0),
        ]),
        in_boundary(claim, "feature2", "feature8", [
            (0, 100),
            (760, 80),
            (0, 35),
        ]),
        in_boundary(claim, "feature2", "feature10", [
            (580, 55),
            (200, 180),
            (1000, 180),
            (1000, 55),
        ]),
        in_boundary(claim, "feature2", "feature12", [
            (580, 55),
            (200, 180),
            (1000, 180),
            (1000, 55),
        ]),
        in_boundary(claim, "feature6", "feature8", [
            (1, 10),
            (1, 100),
            (9, 100),
            (9, 10),
        ]),
        in_boundary(claim, "feature12", "feature17", [
            (19, 0),
            (19, 25),
            (30, 25),
            (30, 0),
        ]),
        in_boundary(claim, "rowIndex", "feature2", [
            (0, 600),
            (0, 1000),
            (80000, 1000),
            (80000, 600),
        ]),
        in_boundary(claim, "rowIndex", "feature10", [
            (0, 110),
            (0, 300),
            (80000, 300),
            (80000, 110),
        ]),
        in_boundary(claim, "rowIndex", "feature12", [
            (0, 17.5),
            (0, 30),
            (80000, 30),
            (80000, 17.5),
        ]),
        in_boundary(claim, "rowIndex", "feature15", [
            (0, -0.1),
            (0, 2.5),
            (80000, 2.5),
            (80000, -0.1),
        ]),
        in_boundary(claim, "rowIndex", "feature17", [
            (0, -0.1),
            (0, 2.5),
            (80000, 2.5),
            (80000, -0.1),
        ]),
    ])

def train_boundary_1(train_features, train_label, model, model_columns=None):
    model_wrapper = ModelInductionWrapper(
        model,
        predicate_accept=_should_accept_claim,
        predicate_reject=_should_reject_claim_1,
        model_columns=model_columns,
    )

    model_wrapper.fit(train_features, train_label)

    return model_wrapper

def train_boundary_2(train_features, train_label, model, model_columns=None):
    model_wrapper = ModelInductionWrapper(
        model,
        predicate_accept=_should_accept_claim,
        predicate_reject=_should_reject_claim_2,
        model_columns=model_columns,
    )

    model_wrapper.fit(train_features, train_label)

    return model_wrapper
