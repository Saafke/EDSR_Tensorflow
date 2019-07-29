# EDSR_Tensorflow

TensorFlow implementation of [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921.pdf)[1].

It was trained on the [Div2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) - Train Data (HR images).

## Requirements
- tensorflow
- numpy
- cv2

# Running

Download [Div2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/). If you want to use another dataset, you will have to calculate the mean of that dataset, and set the new mean in 'main.py'. Code for calculating the mean can be found in data_utils.py.

Train:
- from scratch
`python main.py --train --fromscratch --scale <scale> --traindir /path-to-train-images/`

- resume/load previous
`python main.py --train --scale <scale> --traindir /path-to-train-images/`

Test:
`python main.py --test --scale <scale> --image /path-to-image/`

Export to .pb
`python main.py --export --scale <scale>`

Extra arguments (Nr of resblocks, filters, batch, lr etc.)
`python main.py --help`

## Example
(1) Original picture\
(2) Input image\
(3) Bicubic scaled (3x) image\
(4) EDSR scaled (3x) image\
![Alt text](images/original.png?raw=true "Original picture")
![Alt text](images/input.png?raw=true "Input image picture")
![Alt text](images/bicubicOutput.png?raw=true "Bicubic picture")
![Alt text](images/edsrOutput.png?raw=true "EDSR picture")


## References
[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
