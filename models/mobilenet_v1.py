from tensorflow import keras
layers = keras.layers

def construct(input_shape):
    input_tensor = layers.Input(shape = input_shape, name = 'image')
    net = keras.applications.mobilenet.MobileNet(input_tensor=input_tensor, include_top=False, weights='imagenet', pooling='avg')
    x = net.outputs[0]
    
    # TODO define output layers
    outputs = x

    # Build model
    model = keras.Model(net.inputs, outputs)
    return model

# Simple test function
if __name__ == '__main__':
    construct((224, 224, 3)).summary()