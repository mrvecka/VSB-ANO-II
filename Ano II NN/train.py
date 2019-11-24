import loader
import tensorflow as tf
import ImageClasificationNeuralNetwork as icnn
import numpy as np
import config as cfg

def start_train():
    ld = loader.Loader(cfg.PARKING_GEOMETRY_PATH,
                       [cfg.TRAIN_IMAGES_FREE, cfg.TRAIN_IMAGES_FULL], 0)
    # train_images, train_labels = ld.create_dataset()
    ld.create_dataset()

    image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE], name='input_image_placeholder')
    labels_placeholder = tf.placeholder(tf.float32, [None, cfg.NUM_CLASSES], name='input_label_placeholder')
    is_training = tf.placeholder(tf.bool, name='input_is_training_placeholder')
    hold_prob = tf.placeholder(tf.float32, name='input_hold_prob_placeholder')

    y_predicted = icnn.create_classification_network(image_placeholder, hold_prob, is_training)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predicted, labels=labels_placeholder))

    # add an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE).minimize(cross_entropy)

    # define an accuracy assessment operation
    output_predict = tf.argmax(y_predicted, 1, name='output_prediction')
    correct_prediction = tf.equal(tf.argmax(labels_placeholder, 1), output_predict)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        # initialise the variables

        session.run(init)
        for epoch in range(cfg.EPOCHS):
            for i in range(cfg.ITERATIONS):
                image_batch, labels_batch = ld.get_train_data(cfg.BATCH_SIZE)
                # image_batch = (image_batch - 128) / 128
                session.run(optimizer, 
                                feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: True, hold_prob:cfg.TRAIN_HOLD_PROBABILITY})

            image_batch, labels_batch = ld.get_train_data(cfg.BATCH_SIZE)
            # image_batch = (image_batch - 128) / 128
            test_acc = session.run(accuracy, 
                            feed_dict={image_placeholder: image_batch, labels_placeholder: labels_batch, is_training: False, hold_prob:cfg.TEST_HOLD_PROBABILITY})

            print("Epoch:", (epoch + 1), "test accuracy: {:3f}".format(test_acc*100))


        saver.save(session, cfg.MODEL_SAVE_PATH)
               
if __name__ == "__main__":
    start_train()