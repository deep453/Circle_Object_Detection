import os
import ast
import cv2
import pandas as pd
import tensorflow as tf
from pathlib import Path


def generate_label_map(output_path: Path):
    """Creates label_map.pbtxt required by TensorFlow Object Detection API."""
    label_map_content = """item {\n  id: 1\n  name: 'circle'\n}\n"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(label_map_content)
    print(f"[INFO] Label map created at: {output_path}")


def create_tf_example(row: pd.Series, img_dir: Path) -> tf.train.Example | None:
    """Converts a single dataset record into a TF Example with normalized coordinates."""
    img_name = row["img_name"]
    img_path = img_dir / img_name

    if not img_path.exists():
        print(f"[WARNING] Image missing: {img_path}")
        return None

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[WARNING] Unable to read image file: {img_path}")
        return None

    height, width, _ = image.shape

    # Parse string tuple e.g. "(334, 295)" -> (xmin, ymin)
    box_corner = ast.literal_eval(str(row["box_corner"]))
    xmin, ymin = float(box_corner[0]), float(box_corner[1])
    box_length = float(row["box_length"])
    box_width = float(row["box_width"])

    xmax = xmin + box_length
    ymax = ymin + box_width

    # Normalize bounding box coordinates to [0.0, 1.0]
    xmin_norm = max(0.0, min(1.0, xmin / width))
    xmax_norm = max(0.0, min(1.0, xmax / width))
    ymin_norm = max(0.0, min(1.0, ymin / height))
    ymax_norm = max(0.0, min(1.0, ymax / height))

    with open(img_path, "rb") as f:
        encoded_image = f.read()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "image/height": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[height])
                ),
                "image/width": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[width])
                ),
                "image/filename": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[img_name.encode("utf-8")]
                    )
                ),
                "image/source_id": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=[img_name.encode("utf-8")]
                    )
                ),
                "image/encoded": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[encoded_image])
                ),
                "image/format": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b"png"])
                ),
                "image/object/bbox/xmin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[xmin_norm])
                ),
                "image/object/bbox/xmax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[xmax_norm])
                ),
                "image/object/bbox/ymin": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[ymin_norm])
                ),
                "image/object/bbox/ymax": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[ymax_norm])
                ),
                "image/object/class/text": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[b"circle"])
                ),
                "image/object/class/label": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[1])
                ),
            }
        )
    )
    return example


def main():
    # Define working directory relative paths
    base_dir = Path.cwd()
    dataset_dir = base_dir / "dataset"
    csv_path = dataset_dir / "result.csv"
    img_dir = dataset_dir / "train_image"

    output_record = base_dir / "train.record"
    label_map_path = base_dir / "label_map.pbtxt"

    # Step 1: Create Label Map
    generate_label_map(label_map_path)

    # Step 2: Read CSV annotations
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Processing {len(df)} records from CSV...")

    # Step 3: Write TFRecord file
    valid_count = 0
    with tf.io.TFRecordWriter(str(output_record)) as writer:
        for _, row in df.iterrows():
            tf_example = create_tf_example(row, img_dir)
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())
                valid_count += 1

    print(f"[SUCCESS] {valid_count}/{len(df)} samples written to {output_record}")


if __name__ == "__main__":
    main()