import config as cfg
import tensorflow as tf
import loader
import cv2
import common as com


def start_test():
        
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(cfg.MODEL_SAVE_PATH + '.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('model/'))
        
        graph = tf.get_default_graph()
        image_placeholder = graph.get_tensor_by_name("input_image_placeholder:0")
        is_training = graph.get_tensor_by_name("input_is_training_placeholder:0")
        hold_prob = graph.get_tensor_by_name("input_hold_prob_placeholder:0")
        
        predict = graph.get_tensor_by_name("output_prediction:0")

        
        # load test data, run estimator
        ld_test = loader.Loader(cfg.PARKING_GEOMETRY_PATH, [cfg.TEST_IMAGES_PATH], 1)
        # ground truth data are saved in special order so test images must be loaded in that order
        ld_test.create_test_dataset([1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,3,4,5,6,7,8,9])
        
        predicted = []
        for i in range(ld_test.data_size()):
            test_data = ld_test.trainData[i]
            image = test_data.colored_image
            for j in range(len(test_data.park_lots)):
                park_data = test_data.park_lots[j]
                batch_x = park_data.image
                # batch_x = (batch_x - 128) /128
                park_class = sess.run(predict, feed_dict={image_placeholder:batch_x, is_training: False, hold_prob:cfg.TEST_HOLD_PROBABILITY})
                predicted.append(park_class[0])
                if(park_class[0] == 0):
                    # empty
                    image = cv2.circle(image, (int(park_data.center_x), int(park_data.center_y)), 5, (0, 255, 0), 2) 
                else:
                    # full
                    image = cv2.circle(image, (int(park_data.center_x), int(park_data.center_y)), 5, (0, 0, 255), 2)
            
            # cv2.imshow("predicted", image)            
            # cv2.waitKey(0)
            
        predict_acc = com.compute_accuracy(predicted, cfg.GROUND_TRUTH_PATH)
        print("Final prediction accuracy: ", predict_acc)
            
if __name__ == "__main__":
    start_test()