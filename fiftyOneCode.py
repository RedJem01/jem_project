import fiftyone as fo
import fiftyone.zoo as foz

#Code modified from https://docs.voxel51.com/api/fiftyone.zoo.datasets.base.html#fiftyone.zoo.datasets.base.Kinetics400Dataset
classes = ["brushing teeth"]

dataset = foz.load_zoo_dataset(
    "kinetics-400",
    splits=["train", "test", "validation"],
    classes=classes,
    max_samples=10,
)

session = fo.launch_app(dataset)
