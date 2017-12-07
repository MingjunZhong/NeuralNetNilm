import tensorlayer as tl
import numpy as np
import time



def dict_to_one(dp_dict={}):

    """ Input a dictionary, return a dictionary that all items are
    set to one, use for disable dropout, drop-connect layer and so on.

    Parameters
    ----------
    dp_dict : dictionary keeping probabilities date
    """
    return {x: 1 for x in dp_dict}


def sigmoid(x):

    return 1/(1+np.exp(-x))


def modelsaver(network, path, epoch_identifier=None):

    if epoch_identifier:
        ifile = path + '_' + str(epoch_identifier)+'.npz'
    else:
        ifile = path + '.npz'
    tl.files.save_npz(network.all_params, name=ifile)


def customfit(sess, 
              network, 
              cost, 
              train_op, 
              tra_provider, 
              x, 
              y_, 
              acc=None, 
              n_epoch=50,
              print_freq=1, 
              val_provider=None, 
              save_model=-1, 
              tra_kwag=None, 
              val_kwag=None,
              save_path=None, 
              epoch_identifier=None, 
              earlystopping=True, 
              min_epoch=10,
              patience=10):
    """
        Traing a given network by the given cost function, dataset, n_epoch etc.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        train_op : a TensorFlow optimizer
            like tf.train.AdamOptimizer
        x : placeholder
            for inputs
        y_ : placeholder
            for targets
        acc : the TensorFlow expression of accuracy (or other metric) or None
            if None, would not display the metric
        batch_size : int
            batch size for training and evaluating
        n_epoch : int
            the number of training epochs
        print_freq : int
            display the training information every ``print_freq`` epochs
        X_val : numpy array or None
            the input of validation data
        y_val : numpy array or None
            the target of validation data
        eval_train : boolen
            if X_val and y_val are not None, it refects whether to evaluate the training data
    """
    # parameters for earlystopping
    best_valid = np.inf
    best_valid_acc = np.inf
    best_valid_epoch = min_epoch
    
    # assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"
    print("Start training the network ...")
    start_time_begin = time.time()
    for epoch in range(n_epoch):
        #start_time = time.time()
        loss_ep = 0;
        n_step = 0

        for batch in tra_provider.feed(**tra_kwag):
            X_train_a, y_train_a = batch
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
        loss_ep = loss_ep / n_step

        if epoch >= 0 or (epoch + 1) % print_freq == 0:
            # evaluate the val error at each epoch.
            if val_provider is not None:
                #print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                train_loss, train_acc, n_batch_train = 0, 0, 0
                for batch in tra_provider.feed(**tra_kwag):
                    X_train_a, y_train_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        train_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    train_loss += err;
                    n_batch_train += 1
                #print("   train loss: %f" % (train_loss / n_batch))
                # print (train_loss, n_batch)
                #if acc is not None:
                    #print("   train acc: %f" % (train_acc / n_batch))
                val_loss, val_acc, n_batch_val = 0, 0, 0

                for batch in val_provider.feed(**val_kwag):
                    X_val_a, y_val_a = batch
                    dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err;
                    n_batch_val += 1
                #print("   val loss: %f" % (val_loss / n_batch))
                #if acc is not None:
                    #print("   val acc: %f" % (val_acc / n_batch))
            #else:
                #print("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
        
        if earlystopping:
            if epoch >= min_epoch:
                current_valid = val_loss / n_batch_val
                current_valid_acc = val_acc / n_batch_val
                current_epoch = epoch
                current_train_loss = train_loss / n_batch_train
                current_train_acc = train_acc / n_batch_train
                print('     Current valid loss was {:.6f}, acc was {:.6f}, \
                          train loss was {:.6f}, acc was {:.6f} at epoch {}.'.format(
                      current_valid, current_valid_acc, 
                      current_train_loss, current_train_acc,
                      current_epoch))
                if current_valid < best_valid:
                    best_valid = current_valid
                    best_valid_acc = current_valid_acc
                    best_valid_epoch = current_epoch
                    # save the model parameters
                    modelsaver(network=network, path=save_path, epoch_identifier=None)
                    print('Best valid loss was {:.6f} and acc {:.6f} at epoch {}.'.format(
                          best_valid, best_valid_acc, best_valid_epoch))
                elif best_valid_epoch + patience < current_epoch:
                    print('Early stopping.')
                    print('Best valid loss was {:.6f} and acc {:.6f} at epoch {}.'.format(
                          best_valid, best_valid_acc, best_valid_epoch))  
                    break
                    #raise StopIteration()
                
        else:                
            current_val_loss = val_loss / n_batch_val
            current_val_acc = val_acc / n_batch_val
            current_epoch = epoch
            current_train_loss = train_loss / n_batch_train
            current_train_acc = train_acc / n_batch_train
            print('     Current valid loss was {:.6f}, acc was {:.6f}, \
                          train loss was {:.6f}, acc was {:.6f} at epoch {}.'.format(
                      current_val_loss, current_val_acc, 
                      current_train_loss, current_train_acc,
                      current_epoch))
            # print(save_model > 0, epoch % save_model == 0, epoch/save_model > 0)
            if save_model > 0 and epoch % save_model == 0:
                if epoch_identifier:
                    modelsaver(network=network, path=save_path, epoch_identifier=epoch)
                else:
                    modelsaver(network=network, path=save_path, epoch_identifier=None)
    if not earlystopping:        
        if save_model == -1:
            modelsaver(network=network, path=save_path, epoch_identifier=None)

    print("Total training time: %fs" % (time.time() - start_time_begin))


def customtest(sess, network, acc, test_provider, x, y_, cost, test_kwag=None):
    """
        Test a given non time-series network by the given test data and metric.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        acc : the TensorFlow expression of accuracy (or other metric) or None
            if None, would not display the metric
        X_test : numpy array
            the input of test data
        y_test : numpy array
            the target of test data
        x : placeholder
            for inputs
        y_ : placeholder
            for targets
        batch_size : int or None
            batch size for testing, when dataset is large, we should use minibatche for testing.
            when dataset is small, we can set it to None.
        cost : the TensorFlow expression of cost or None
            if None, would not display the cost
    """
    test_loss, test_acc, n_batch = 0, 0, 0
    for batch in test_provider.feed(**test_kwag):
        X_test_a, y_test_a = batch
        dp_dict = dict_to_one(network.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        if acc is not None:
            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
            test_acc += ac
        else:
            err = sess.run(cost, feed_dict=feed_dict)
        test_loss += err;
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    if acc is not None:
        print("   test acc: %f" % (test_acc / n_batch))



def custompredict(sess, network, output_provider, x, fragment_size=1000, output_length=1, y_op=None, out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """
    dp_dict = dict_to_one(network.all_drop)  # disable noise layers

    if y_op is None:
        y_op = network.outputs
    output_container = []
    gt = []
    banum = 0
    for batch in output_provider.feed(**out_kwag):
        # print banum
        banum += 1
        X_out_a, gt_batch = batch
        # print 'hi', X_out_a.mean()
        fra_num = X_out_a.shape[0] / fragment_size
        offset = X_out_a.shape[0] % fragment_size
        final_output = np.zeros((X_out_a.shape[0], output_length))
        for fragment in xrange(fra_num):
            x_fra = X_out_a[fragment * fragment_size:(fragment + 1) * fragment_size]
            feed_dict = {x: x_fra, }
            feed_dict.update(dp_dict)
            final_output[fragment * fragment_size:(fragment + 1) * fragment_size] = sess.run(y_op, feed_dict=feed_dict).reshape(-1,output_length)

        if offset > 0:
            feed_dict = {x: X_out_a[-offset:], }
            feed_dict.update(dp_dict)
            final_output[-offset:] = sess.run(y_op, feed_dict=feed_dict).reshape(-1,output_length)
        output_container.append(final_output)
        gt.append(gt_batch)
        # print 'hello', final_output.mean()
    return np.vstack(output_container), np.vstack(gt)


def custompredict_add(sess, network, output_provider, x, seqlength, fragment_size=1000, output_length=1,
                      y_op=None, out_kwag=None, std=1, mean=0):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """
    dp_dict = dict_to_one(network.all_drop)  # disable noise layers

    prediction = np.zeros((seqlength))
    if y_op is None:
        y_op = network.outputs

    banum = 0
    datanum = 0
    ave = np.ones((seqlength)) * output_length
    ave[:output_length - 1] = np.arange(1, output_length)
    ave[-(output_length - 1):] = np.arange(output_length - 1, 0, -1)


    for batch in output_provider.feed(**out_kwag):
        # print banum
        banum += 1
        X_out_a, gt_batch = batch
        # print 'hi', X_out_a.mean()
        fra_num = X_out_a.shape[0] / fragment_size
        offset = X_out_a.shape[0] % fragment_size
        # final_output = np.zeros((X_out_a.shape[0], output_length))
        for fragment in xrange(fra_num):
            x_fra = X_out_a[fragment * fragment_size:(fragment + 1) * fragment_size]
            feed_dict = {x: x_fra, }
            feed_dict.update(dp_dict)
            batch_prediction = sess.run(y_op, feed_dict=feed_dict).reshape(-1, output_length) * std + mean
            for sequence in batch_prediction:
                prediction[datanum:datanum + output_length] += sequence
                datanum += 1

        if offset > 0:
            feed_dict = {x: X_out_a[-offset:], }
            feed_dict.update(dp_dict)
            batch_prediction = sess.run(y_op, feed_dict=feed_dict).reshape(-1,output_length)
            for sequence in batch_prediction:
                prediction[datanum:datanum + output_length] += sequence
                datanum += 1


        # print 'hello', final_output.mean()
    return prediction/ave







