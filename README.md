# MobileNet
An implementation of `Synchronous Advantage Actor Critic (A2C)` introduced in TensorFlow. A2C is a variant of advantage actor critic introduced by [OpenAI in their published baselines](https://github.com/openai/baselines). However, these baselines are difficult to understand and modify. So, I implemented the A2C based on their implementation but in a clearer and simpler way.


## Depthwise Separable Convolution
<div align="center">
<img src="https://github.com/MG2033/MobileNet/blob/master/figures/dws.png"><br><br>
</div>

## ReLU6
The paper uses ReLU6 as an activation function. ReLU6 was first introduced in [Convolutional Deep Belief Networks on CIFAR-10](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf) as a ReLU with clipping its output at 6.0.

## Usage
### Main Dependencies
 ```
 tensorflow 1.3.0
 numpy 1.13.1
 tqdm 4.15.0
 bunch 1.0.1
 matplotlib 2.0.2
 ```
### Train and Test
1. Prepare your data, and modify the data_loader.py/DataLoader/load_data() method.
2. Modify the config/test.json to meet your needs.

Note: If you want to test that the model is pretrained and working properly, I've added some test images from different classes in directory 'data/test_images'. All of them are classified correctly.

### Run
```
python main.py config/test.json
```
The file 'test.json' is just an example of a file. If you run it as is, it will test the model against the images in directory 'data/test_images'. You can create your own configuration file for training/testing.

## Benchmarking
The paper has achieved 569 Mult-Adds. In my implementation, I have achieved approximately 1140 MFLOPS. The paper counts multiplication+addition as one unit. My result verifies the paper as roughly dividing 1140 by 2 is equal to 569 unit.

To calculate the FLOPs in TensorFlow, make sure to set the batch size equal to 1, and execute the following line when the model is loaded into memory.
```
tf.profiler.profile(
        tf.get_default_graph(),
        options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')
```
I've already implemented this function. It's called ```calculate_flops()``` in `utils.py`. Use it directly if you want.

## Updates
* Inference and training are working properly.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
