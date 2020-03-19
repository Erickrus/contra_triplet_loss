# -*- coding: utf-8 -*-
import numpy as np
import librosa
import sys
import os
import glob

from PIL import Image
from matplotlib import cm

class VoiceFeature:
    def __init__(self, imageHeight=32):
        self.imageHeight = imageHeight
        self.defaultFps = 25.0

    def _normalize(self, x):
        return (x / max(abs(x.max()), abs(x.min())) + 1.0) / 2.0 

    def extract(self, 
        inputFilename, 
        outputFilename=None, 
        color=True,
        fps = 25.0
    ):
        adjWavFilename = inputFilename[:-4] + ".16k1c.wav"
        os.system("sox %s -c %d -r %dk -t wav %s" % (
            inputFilename, 1, 16, adjWavFilename )
        )

        y, sr = librosa.load(adjWavFilename)
        os.system("rm %s"%adjWavFilename)
        hopLen = int(sr / fps)
        mels = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=self.imageHeight,
            fmax = 8000, # for voice, let's trim exceeding 8000Hz
            n_fft=hopLen*2,
            hop_length=hopLen
        )

        # some of the approaches used in this class referenced following link
        # https://stackoverflow.com/questions/56719138/how-can-i-save-a-librosa-spectrogram-plot-as-a-specific-sized-image/57204349#57204349
        mels = self._normalize(np.log(mels + 1e-9))
        mels = np.flip(mels, axis=0)

        if color:
            # render grayscale to colormap and librosa default cmap
            # https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
            # https://librosa.github.io/librosa/generated/librosa.display.cmap.html
            im = Image.fromarray(np.uint8(cm.magma(mels)*255))
        else:
            im = Image.fromarray(np.uint8(mels*255))

        if outputFilename != None:
            im.convert("RGB").save(outputFilename)
        else:
            return im

if __name__ == "__main__":
    vf = VoiceFeature()
    for filename in glob.glob("/f/data/dataset/thchs30/data_thchs30/train/*.wav"):
        outputFilename = filename[:-4] + ".png"
        print(filename, outputFilename)
        vf.extract(filename, color=True, outputFilename=outputFilename)
