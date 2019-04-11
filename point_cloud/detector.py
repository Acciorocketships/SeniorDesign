from imageai.Detection import ObjectDetection
import sys
import os

class Detector:
    def __init__(self):
        self.execution_path = os.getcwd()
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath( os.path.join(self.execution_path , "resnet50_coco_best_v2.0.1.h5"))
        self.detector.loadModel()

    def run(self, image_matrix):
        detections = self.detector.detectObjectsFromImage(input_type="array", input_image=image_matrix)
        results = []
        for eachObject in detections:
            pts = eachObject["box_points"]
            results.append(pts)
        return results

    def run_save(self, image_path, storage_path='detect_target.txt'):
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path , image_path))
        with open(storage_path, "w+") as f:
            for eachObject in detections:
                print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"]  )
                pts = eachObject["box_points"]
                f.write(str(pts[0]) + ' ' + str(pts[1]) + ' ' + str(pts[2]) + ' ' + str(pts[3]) + "\n")

    def run_return(self, image_path):
        detections = self.detector.detectObjectsFromImage(input_image=os.path.join(self.execution_path , image_path))
        result = []
        for eachObject in detections:
            pts = eachObject["box_points"]
            temp = []
            temp.append(pts[0])
            temp.append(pts[1])
            temp.append(pts[2])
            temp.append(pts[3])
            result.append(temp)
        return result

d = Detector()
d.run_save(sys.argv[1])