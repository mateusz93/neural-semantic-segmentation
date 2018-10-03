"""A script to predict a video of frames using Epistemic Tiramisu."""
import sys
import imageio
# imageio.plugins.ffmpeg.download()
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import transform
from matplotlib import pyplot as plt
from src import camvid
from src import tiramisu
from src.utils import heatmap


if __name__ == '__main__':
    # load the mapping of labels for CamVid
    mapping = pd.read_table('11_class.txt',
        sep=r'\s+',
        names=['og', 'new'],
        index_col='og'
    )['new'].to_dict()
    camvid11 = camvid.CamVid(mapping=mapping)
    # the video file to predict
    video_file = sys.argv[1]
    out_file = video_file.split('.')[0] + '_pred.mp4'

    # create a stream to read the input video
    reader = imageio.get_reader(video_file)
    # the shape to resize frames to
    image_shape = (352, 480, 3)
    # build the model for the image shape and number of labels
    model = tiramisu.build_epi_approx_tiramisu(image_shape, camvid11.n)
    model.layers[1].load_weights('models/Tiramisu103-CamVid11-fine.h5')


    def predict(frame) -> tuple:
        """
        Return post-processed predictions for the given frame.

        Args:
            frame: the frame to predict

        Returns:
            a tuple of for NumPy tensors with RGB data:
            - the unmapped RGB predicted mean values from the model
            - the meatmap RGB values of the model variance

        """
        # predict mean values and variance
        X = frame[None, ...]
        u, s2 = model.predict(X)
        # normalize the uncertainty
        s2 = plt.Normalize()(s2)
        # return X values, unmapped u values, and heatmap of s2
        return camvid11.unmap(u)[0], heatmap(s2, 'afmhot')[0]

    # get the FPS of the source video
    fps = reader.get_meta_data()['fps']
    # create a video writer with source FPS
    writer = imageio.get_writer(out_file, fps=fps)

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
        u, s2 = predict(X)
        h, w, c = image_shape
        # convert the three images into a singular image (side-by-side)
        image = np.zeros((h, w * 3, c), dtype='uint8')
        for idx, piece in enumerate([X, u, s2]):
            image[:, idx*w:(idx+1)*w, :] = piece
        # save the image to the stream
        writer.append_data(image)
    # close the writer
    writer.close()
    # close the reader
    reader.close()
