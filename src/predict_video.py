"""A script to predict a video of frames using Epistemic Tiramisu."""
import imageio
import numpy as np
from tqdm import tqdm
from skimage import transform


def predict_video(
    video_path: str,
    out_path: str,
    camvid: 'CamVid',
    model: 'Model',
    predict: 'Callable',
) -> None:
    """
    Predict a video stream and stream the predictions to disk.

    Args:
        video_path: the path to the video to stream
        out_path: the path to write the output video stream to
        camvid: the CamVid instance for un-mapping segmentations
        model: the model to generate predictions and uncertainty from
        predict: the predict method for the given model

    Returns:
        None

    """
    # create a stream to read the input video
    reader = imageio.get_reader(video_path)
    # the shape to resize frames to
    # image_shape = (352, 480, 3)
    image_shape = model.input_shape
    # create a video writer with source FPS
    writer = imageio.get_writer(out_path, fps=reader.get_meta_data()['fps'])

    # iterate over the frames in the source video
    for frame in tqdm(reader):
        # resize the image to the acceptable size for the model
        X = transform.resize(frame, image_shape,
            anti_aliasing=False,
            mode='symmetric',
            clip=False,
            preserve_range=True,
        )
        # predict mean and model variance of the frame
        outs = predict(model, X, camvid)
        h, w, c = image_shape
        # convert the three images into a singular image (side-by-side)
        image = np.zeros((h, w * len(outs), c), dtype='uint8')
        for idx, piece in enumerate(outs):
            image[:, idx*w:(idx+1)*w, :] = piece
        # save the image to the stream
        writer.append_data(image)

    # close the writer
    writer.close()
    # close the reader
    reader.close()


# explicitly define the outward facing API of this module
__all__ = [predict_video.__name__]
