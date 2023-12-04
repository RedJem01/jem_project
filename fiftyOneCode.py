import fiftyone as fo
import fiftyone.zoo as foz


name = "project_dataset"

dataset_dir = "C:/Users/jemst/Documents/Uni/Year_3/Project/project_dataset"

dataset_type = fo.types.ImageClassificationDirectoryTree

classes = ['brushing_teeth', 'cutting_nails', 'doing_laundry', 'folding_clothes', 'washing_dishes']

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)

session = fo.launch_app(dataset)
