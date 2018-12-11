import scipy.stats as stats
from config.global_parameters import number_of_classes
import numpy as np

from keras import backend as K
from keras.engine.topology import Layer

class _Outer(Layer):

    """ Not using """
    def __init__(self, input_dim=64, **kwargs):
        self.input_dim = input_dim   #d
        if self.input_dim:
          kwargs['input_shape'] = (self.input_dim,)
        super(Outer, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim}
        base_config = super(Outer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        return

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) < 2:
          raise Exception('BilinearTensorLayer must be called on a list of tensors '
                          '(at least 2). Got: ' + str(inputs))
        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        outer_product = e1[:,:,np.newaxis] * e2[:,np.newaxis,:]
#        outer_product = K.squeeze(outer_product, -1)
        outer_product = K.reshape(outer_product,(batch_size, -1))
        return outer_product

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, self.input_dim**2)

class _BilinearTensorLayer(Layer):

    def __init__(self, output_dim, input_dim=None, **kwargs):
        self.output_dim = output_dim
        self.input_dim = input_dim

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, )
        super(BilinearTensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean, stddev = 0.0, 1.0
        k = self.output_dim
        d = self.input_dim

        W_init = stats.truncnorm.rvs(-2*stddev, 2*stddev, loc=mean, scale=stddev, size=(k,d,d))
        V_init = stats.truncnorm.rvs(-2*stddev, 2*stddev, loc=mean, scale=stddev, size=(2*d,k))

        self.W = K.variable(W_init)
        self.V = K.variable(V_init)
        self.b = K.zeros((self.input_dim, ))
        self.trainable_weights = [self.W, self.V, self.b]

    def call(self, inputs, mask=None):

        assert (len(inputs) == 2), "You must pass only two tensor inputs"

        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim
        feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
        bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1) for i in range(k)]
        result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)

        return result

    def get_output_shape_for(self, input_shape):
        #print 'is:', input_shape
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)


class BilinearTensorLayer(Layer):

    def __init__(self, input_dim=None, output_dim=number_of_classes, **kwargs):
        """ output dim = number of classes """
        self.output_dim = output_dim 
        self.input_dim = input_dim

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, )
        super(BilinearTensorLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        mean, stddev = 0.0, 1.0
        k = self.output_dim
        d = self.input_dim

        W_init = stats.truncnorm.rvs(-2*stddev, 2*stddev, loc=mean, scale=stddev, size=(k,d,d))

        self.W = K.variable(W_init)
        self.b = K.zeros((self.input_dim, ))
        self.trainable_weights = [self.W, self.b]

    def call(self, inputs, mask=None):

        assert (len(inputs) == 2), "You must pass only two tensor inputs"

        e1 = inputs[0]
        e2 = inputs[1]
        batch_size = K.shape(e1)[0]
        k = self.output_dim
        bilinear_tensor_products = [K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1) for i in range(k)]
        result = K.sigmoid(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)))

        return result

    def get_output_shape_for(self, input_shape):
        #print 'is:', input_shape
        batch_size = input_shape[0][0]
        return (batch_size, self.output_dim)
