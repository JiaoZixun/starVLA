import numpy as np
from PIL import Image

from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import (
    resize_image_like_training,
)


def test_libero_eval_resize_matches_training_loader_filter():
    frame = np.random.default_rng(7).integers(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )

    eval_frame = resize_image_like_training(frame, (224, 224))
    train_frame = np.asarray(Image.fromarray(frame).resize((224, 224)))

    np.testing.assert_array_equal(eval_frame, train_frame)


def test_libero_eval_resize_preserves_frames_already_at_training_size():
    frame = np.random.default_rng(11).integers(
        0, 256, size=(224, 224, 3), dtype=np.uint8
    )

    eval_frame = resize_image_like_training(frame, (224, 224))

    assert eval_frame is frame
