# Lip-Motion-Recognition
This is graduation project of Konkuk Univ.

## Model architecture
<img src='https://raw.githubusercontent.com/johun204/lip-motion-recognition/main/media/model.png'>

The difference image of consecutive frames of video was used as input.

## Result
<img src='https://raw.githubusercontent.com/johun204/lip-motion-recognition/main/media/result.gif' height='250px'> <img src='https://raw.githubusercontent.com/johun204/lip-motion-recognition/main/media/result2.gif' height='250px'>

## Requirements

* Python >= 3.5
* Pytorch 0.4.0
* CUDA (if CUDA available)

## Dataset
* [Ouluvs2](http://www.ee.oulu.fi/research/imag/OuluVS2/) (Phase 2)

  the subject was asked to speak ten daily-use short English phrases. The same set of phrases was used in our previously collected OuluVS database. Every phrase was uttered three times.
  
## Data processing flow-chart
<img src='https://raw.githubusercontent.com/johun204/lip-motion-recognition/main/media/data_processing.png' height='300px'>


## Accuracy (each frame)
* "Excuse me" : 63.50% [8271 / 13024]
* "Goodbye" : 86.46% [7429 / 8592]
* "Hello" : 32.32% [1986 / 6144]
* "How are you" (3) : 59.73% [6279 / 10512]
* "Nice to meet you" (4) : 63.13% [9253 / 14656]
* "See you" (5) : 58.00% [4687 / 8080]
* "I am sorry" (6) : 50.24% [5507 / 10960]
* "Thank you" (7) : 61.42% [4973 / 8096]
* "Have a good time" (8) : 68.83% [9791 / 14224]
* "You are welcome" (9) : 51.81% [6641 / 12816]

 Total Accuracy : 76.67% 
