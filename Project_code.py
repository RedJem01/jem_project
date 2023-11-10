import fiftyone as fo
import fiftyone.zoo as foz

classes = ["brushing teeth"]

dataset = foz.load_zoo_dataset(
    "kinetics-400",
    splits=["train", "test", "validation"],
    classes=classes,
    max_samples=10,
)

session = fo.launch_app(dataset)
