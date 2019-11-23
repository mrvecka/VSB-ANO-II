import cv2
import numpy as np
import random

class TrainData:
    
    def __init__(self):
        self.image = []
        self.label = []
        self.center_x = 0
        self.center_y = 0

class TestData:
    def __init__(self):
        self.image = []
        self.colored_image = []
        self.park_lots = []

class Loader:
    
    def __init__(self, fov_path, img_paths,start_from):
        self.fov_file = fov_path
        self.img_paths = img_paths
        self.fovs = []
        self.trainData = []
        self.start_from = start_from
        
    def load_FOVs(self):
        with open(self.fov_file, 'r') as infile_label:

            for line in infile_label:
                line = line.rstrip('\n')
                if "->" not in line:
                    continue
                start = line.index("->") +2
                str_data = line[start:]
                str_data = str_data.rstrip(";")
                data = np.asarray([x.split(',') for x in str_data.split(';')]).flatten()
                
                fov = [[int(data[0]),int(data[1])],[int(data[2]),int(data[3])],[int(data[4]),int(data[5])],[int(data[6]),int(data[7])]]
                self.fovs.append(fov)
    def create_train_images(self):
        space_size = (80,80)
        for j in range(0,len(self.img_paths)):
            
            for m in range(self.start_from,500):
                path = self.img_paths[j] + r'\test' + str(m) + r'.jpg'
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    break
                label = [1,0]
                if "full" in self.img_paths[j]:
                    label = [0,1]
            
                for i in range(len(self.fovs)):
                    fov = self.fovs[i]
                    
                    dest_fov = np.array(fov,dtype=np.float32).reshape(4,2)
                    dest_fov[0][0] = 0
                    dest_fov[0][1] = 0
                    dest_fov[1][0] = space_size[1]
                    dest_fov[1][1] = 0
                    dest_fov[2][0] = space_size[1]
                    dest_fov[2][1] = space_size[0]
                    dest_fov[3][0] = 0
                    dest_fov[3][1] = space_size[0]

                    M = cv2.getPerspectiveTransform(np.array(fov,dtype=np.float32).reshape(4,2), dest_fov)
                    result = cv2.warpPerspective(img, M, space_size)
                    
                    # cv2.imshow("region of interest", result)
                    # cv2.waitKey(0)
                    data = TrainData()
                    data.image = result
                    data.label = label
                    self.trainData.append(data)
                    
    def create_test_images(self,image_order):
        space_size = (80,80)
        for j in range(0,len(self.img_paths)):
            
            for m in range(len(image_order)):
                path = self.img_paths[j] + r'\test' + str(image_order[m]) + r'.jpg'
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    break
            
                data = TestData()
                data.image = img
                data.colored_image = cv2.imread(path, cv2.IMREAD_COLOR)
                for i in range(len(self.fovs)):
                    fov = self.fovs[i]
                    
                    dest_fov = np.array(fov,dtype=np.float32).reshape(4,2)
                    dest_fov[0][0] = 0
                    dest_fov[0][1] = 0
                    dest_fov[1][0] = space_size[1]
                    dest_fov[1][1] = 0
                    dest_fov[2][0] = space_size[1]
                    dest_fov[2][1] = space_size[0]
                    dest_fov[3][0] = 0
                    dest_fov[3][1] = space_size[0]

                    M = cv2.getPerspectiveTransform(np.array(fov,dtype=np.float32).reshape(4,2), dest_fov)
                    result = cv2.warpPerspective(img, M, space_size)
                    
                    # cv2.imshow("region of interest", result)
                    # cv2.waitKey(0)
                    park = TrainData()
                    park.image = np.reshape(result, (-1,80,80))
                    park.center_x = fov[2][0] - (fov[2][0] - fov[0][0]) / 2
                    park.center_y = fov[2][1] - (fov[2][1] - fov[0][1]) / 2
                    data.park_lots.append(park)
                
                
                self.trainData.append(data)
            
    def create_dataset(self):
        self.load_FOVs()
        self.create_train_images()
        # for i in range(10):
        #     data = random.sample(self.trainData, 1)
            
            # cv2.imshow("test",data[0].image)
            # cv2.waitKey(0)
            
        #return self.trainData, self.trainData
        
    def create_test_dataset(self,image_order):
        self.load_FOVs()
        self.create_test_images(image_order)
        
    def data_size(self):
        return len(self.trainData)
    
    def get_train_data(self,batch_size):
        data = random.sample(self.trainData, batch_size)
        result_image = []
        result_label = []
        for x in range(batch_size):
            result_image.append(data[x].image)
            result_label.append(data[x].label)

        return np.asarray(result_image), np.asarray(result_label)
    
    def create_fake_data(self, batch_size):
        images = np.random.rand(batch_size,80,80)
        labels = np.zeros((batch_size,2))
        labels[:,1] = 1
        # print(labels)
        return np.asarray(images), np.asarray(labels)
    
if __name__ == "__main__":
    ld = Loader(r'D:\Skola\9.semester\ANO II\Ano II NN\parking_map.txt',
                       [r'D:\Skola\9.semester\ANO II\Ano II NN\train_images\free',r'D:\Skola\9.semester\ANO II\Ano II NN\train_images\full'],0)
    
    ld.create_dataset()
    img, lbl = ld.get_train_data(8)
    
    for i in range(8):
        cv2.imshow("test",img[i])
        cv2.waitKey(0)
        