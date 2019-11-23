import loader
import tensorflow as tf
import ImageClasificationNeuralNetwork as icnn
import cv2
import numpy as np
import config as cfg


def compute_accuracy(predicted, ground_truth_path):
    
    ground = []
    with open(ground_truth_path, 'r') as infile_label:

        for line in infile_label:
            line = line.rstrip('\n')
            ground.append(int(line))
           
    # first aproach
    falsePositive = 0
    falseNegative = 0
    truePositive = 0
    trueNegative = 0
    
    for i in range(len(predicted)):
        if (predicted[i] == 1 and ground[i] == 0):
            falsePositive +=1
            
        if (predicted[i] == 0 and ground[i] == 1):
            falseNegative +=1
            
        if (predicted[i] == 1 and ground[i] == 1):
            truePositive +=1
            
        if (predicted[i] == 0 and ground[i] == 0):
            trueNegative +=1  
        
    accuracy_1 = (float(truePositive + trueNegative) / float(truePositive + trueNegative + falsePositive + falseNegative))*100
    print("Accuracy according to teacher: ", accuracy_1)

    # second aproach
    equals = [] 
    for i in range(0,len(ground)):
        if(ground[i] == predicted[i]):
            equals.append(1)
        else:
            equals.append(0)
            
    non_zero = np.count_nonzero(equals)
    accuracy_2 = non_zero / len(equals)
    return accuracy_2 *100

def start_train():
    ld = loader.Loader(cfg.PARKING_GEOMETRY_PATH,
                       [cfg.TRAIN_IMAGES_FREE, cfg.TRAIN_IMAGES_FULL], 0)
    # train_images, train_labels = ld.create_dataset()
    ld.create_dataset()

    image_placeholder = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE])
    labels_placeholder = tf.placeholder(tf.float32, [None, cfg.NUM_CLASSES])
    is_training = tf.placeholder(tf.bool)
    hold_prob = tf.placeholder(tf.float32)

    y_predicted = icnn.create_classification_network(image_placeholder, hold_prob, is_training)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predicted, labels=labels_placeholder))

    # add an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=cfg.LEARNING_RATE).minimize(cross_entropy)

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(labels_placeholder, 1), tf.argmax(y_predicted, 1))
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


        path = saver.save(session, cfg.MODEL_SAVE_PATH)
        
        # load test data, run estimator
        ld_test = loader.Loader(cfg.PARKING_GEOMETRY_PATH, [cfg.TEST_IMAGES_PATH], 1)
        # ground truth data are saved in special order so test images must be loaded in that order
        ld_test.create_test_dataset([1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,3,4,5,6,7,8,9])
        
        predicted = []
        predict = tf.argmax(y_predicted, 1)     
        for i in range(ld_test.data_size()):
            test_data = ld_test.trainData[i]
            image = test_data.colored_image
            for j in range(len(test_data.park_lots)):
                park_data = test_data.park_lots[j]
                batch_x = park_data.image
                # batch_x = (batch_x - 128) /128
                park_class = session.run(predict, feed_dict={image_placeholder:batch_x, is_training: False, hold_prob:cfg.TEST_HOLD_PROBABILITY})
                predicted.append(park_class[0])
                if(park_class[0] == 0):
                    # empty
                    image = cv2.circle(image, (int(park_data.center_x), int(park_data.center_y)), 5, (0, 255, 0), 2) 
                else:
                    # full
                    image = cv2.circle(image, (int(park_data.center_x), int(park_data.center_y)), 5, (0, 0, 255), 2)
            
            # result_path = r'D:\Skola\9.semester\ANO II\Ano II NN\result\image'+str(i)+r'.png'
            # cv2.imwrite(result_path, image) 
            # cv2.imshow("predicted", image)            
            # cv2.waitKey(0)
                          
        predict_acc = compute_accuracy(predicted, cfg.GROUND_TRUTH_PATH)
        print("Final prediction accuracy: ", predict_acc)

            
if __name__ == "__main__":
    start_train()