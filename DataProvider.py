import numpy as np


class DoubleSourceSlider(object):

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets, flatten=True):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        max_batchsize = inputs.size - 2 * self.offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            if flatten:
                yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                      targets[excerpt + self.offset]
            else:
                yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                      targets[excerpt + self.offset].reshape(-1, 1)

    def generate_test_data(self, inputs, targets, targets_gt, offset, flatten=True):

            shuffle = False
            inputs, targets = inputs.flatten(), targets.flatten()
            assert inputs.size == targets.size
            max_batchsize = inputs.size - 2 * offset
            batchsize = max_batchsize
            #if self.batchsize < 0:
            #    self.batchsize = max_batchsize
    
            indices = np.arange(max_batchsize)
            if shuffle:
                np.random.shuffle(indices)
    
            for start_idx in range(0, max_batchsize, batchsize):
                excerpt = indices[start_idx:start_idx + batchsize]
                if flatten:
                    yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
                          targets[excerpt + offset], \
                            targets_gt[excerpt + offset]
                else:
                    yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
                          targets[excerpt + offset].reshape(-1, 1), \
                            targets_gt[excerpt + offset].reshape(-1, 1)


class DoubleSourceSlider(object):
    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets, flatten=True):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size
        max_batchsize = inputs.size - 2 * self.offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            if flatten:
                yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                      targets[excerpt + self.offset]
            else:
                yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                      targets[excerpt + self.offset].reshape(-1, 1)


class S2S_Slider(object):

    def __init__(self, batchsize, shuffle, length):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.length = length

    def feed(self, inputs, targets, flatten=True):

        inputs, targets = inputs.flatten(), targets.flatten()
        assert inputs.size == targets.size

        max_batchsize = inputs.size - self.length
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            
            yield np.array([inputs[idx:idx + self.length] for idx in excerpt]), \
                  np.array([targets[idx:idx + self.length] for idx in excerpt])

    # def generate_test_data(self, inputs, targets, targets_gt, offset, flatten=True):
    #
    #     shuffle = False
    #     inputs, targets = inputs.flatten(), targets.flatten()
    #     assert inputs.size == targets.size
    #     max_batchsize = inputs.size - 2 * offset
    #     batchsize = max_batchsize
    #     # if self.batchsize < 0:
    #     #    self.batchsize = max_batchsize
    #
    #     indices = np.arange(max_batchsize)
    #     if shuffle:
    #         np.random.shuffle(indices)
    #
    #     for start_idx in range(0, max_batchsize, batchsize):
    #         excerpt = indices[start_idx:start_idx + batchsize]
    #         if flatten:
    #             yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
    #                   targets[excerpt + offset], \
    #                   targets_gt[excerpt + offset]
    #         else:
    #             yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
    #                   targets[excerpt + offset].reshape(-1, 1), \
    #                   targets_gt[excerpt + offset].reshape(-1, 1)


class MultiApp_Slider(object):

    def __init__(self, batchsize, shuffle, offset):

        self.batchsize = batchsize
        self.shuffle = shuffle
        self.offset = offset

    def feed(self, inputs, targets, flatten=True):

        # inputs, targets = inputs.flatten(), targets.flatten()
        inputs = inputs.flatten()

        assert inputs.shape[0] == targets.shape[0]
        max_batchsize = inputs.size - 2 * self.offset
        if self.batchsize < 0:
            self.batchsize = max_batchsize

        indices = np.arange(max_batchsize)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, max_batchsize, self.batchsize):
            excerpt = indices[start_idx:start_idx + self.batchsize]
            # if flatten:
            #     yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
            #           targets[excerpt + self.offset]
            # else:
            yield np.array([inputs[idx:idx + 2 * self.offset + 1] for idx in excerpt]), \
                  targets[excerpt + self.offset, :]

    # def generate_test_data(self, inputs, targets, targets_gt, offset, flatten=True):
    #
    #     shuffle = False
    #     inputs, targets = inputs.flatten(), targets.flatten()
    #     assert inputs.size == targets.size
    #     max_batchsize = inputs.size - inputs.size - self.length
    #     batchsize = max_batchsize
    #     # if self.batchsize < 0:
    #     #    self.batchsize = max_batchsize
    #
    #     indices = np.arange(max_batchsize)
    #     if shuffle:
    #         np.random.shuffle(indices)
    #
    #     for start_idx in range(0, max_batchsize, batchsize):
    #         excerpt = indices[start_idx:start_idx + batchsize]
    #         if flatten:
    #             yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
    #                   targets[excerpt + offset], \
    #                   targets_gt[excerpt + offset]
    #         else:
    #             yield np.array([inputs[idx:idx + 2 * offset + 1] for idx in excerpt]), \
    #                   targets[excerpt + offset].reshape(-1, 1), \
    #                   targets_gt[excerpt + offset].reshape(-1, 1)




class DoubleSourceProvider(object):

    def __init__(self, batchsize, shuffle):

        self.batchsize = batchsize
        self.shuffle = shuffle

    def feed(self, inputs, targets):
        assert len(inputs) == len(targets)
        if self.batchsize == -1:
            self.batchsize = len(inputs)
        if self.shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batchsize + 1, self.batchsize):
            if self.shuffle:
                excerpt = indices[start_idx:start_idx + self.batchsize]
            else:
                excerpt = slice(start_idx, start_idx + self.batchsize)
            yield inputs[excerpt], targets[excerpt]


class Transformer(object):
    
    def __init__(self, mu, norm):
        
        self.mu = mu
        self.norm = norm

    def MuLawQuantisation(self, data, quantization=True):
        """
        Perform the mu-law transformation
        ------------------------------------------
        :arg
        data: data that needs to be transform
        mu: scale
        norm: normalisation constant
        quantization: quantize to integral, default True

        :return
        The transformed data

        """ 
        data = data.flatten()
        data = data/self.norm

        mu_law = np.sign(data)*(np.log(1+self.mu*np.abs(data))/np.log(1+self.mu))*self.mu

        if quantization:
            return np.round(mu_law)
        else:
            return mu_law

    def InverseMuLaw(self, data, sample=False):
        """
        Perform the inverse mu-law transformation
        --------------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant

        :return
        The inverse transformed data
        """     
        if sample:        
            means = data.flatten()
            cov = np.eye(data.size)*sample        
            data = np.random.multivariate_normal(means, cov)

        self.mu = float(self.mu)
        data /= self.mu

        recover = np.sign(data)*(1/self.mu)*((1+self.mu)**np.abs(data)-1)

        return recover*self.norm
    
    def LinearQuantisation(self, data, quantization=True):
        """
        Perform the linear quantisation.
        --------------------------------------
        :arg
        data: data that needs to be transform
        mu: scale
        norm: normalisation constant
        quantization: quantize to integral, default True

        :return
        The transformed data

        """ 
        data = data.flatten()
        gap = int(self.norm/self.mu)

        if quantization:
            return np.round(data/gap)
        else:
            return data/gap
        
    def InverseLinear(self, data):
        """
        Perform the inverse linear quantisation.
        -------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant

        :return
        The transformed data
        """     
        return data*int(self.norm/self.mu)
    
    def Normalise(self, data):
        """
        Perform the normalisation (data-mu)/norm.
        --------------------------------------
        :arg
        data: data that needs to be transformed
        mu: scale
        norm: normlisation constant

        :return
        The normalized data

        """     
        return (data-self.mu)/self.norm
    
    def InverseNormalise(self, data):
        """
        Perform the in-normalisation data*norm+mu.
        ------------------------------------------------
        :arg
        data: data that needs to be inverse-transformed
        mu: scale
        norm: normalisation constant
        :return
        The in-normalized data

        """  
        
        return data*self.norm+self.mu




