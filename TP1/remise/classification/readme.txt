highway : python .\yolo.py --path data/highway/input --frames 1700 --output highway --yolo yolo-coco --groundtruth data/highway/groundtruth --gtfrom 470 --gtto 1700

canoe : python .\yolo.py --path data/canoe/input --frames 1189 --output canoe --yolo yolo-coco --groundtruth data/canoe/groundtruth --gtfrom 800 --gtto 1189

streetCornerAtNight : python .\yolo.py --path data/streetCornerAtNight/input --frames 5200 --output streetCornerAtNight --yolo yolo-coco --groundtruth data/streetCornerAtNight/groundtruth --gtfrom 800 --gtto 2999

turnpike : python .\yolo.py --path data/turnpike/input --frames 1500 --output turnpike --yolo yolo-coco --groundtruth data/turnpike/groundtruth --gtfrom 800 --gtto 1149

---------------------------------------------------------------------------------
Download yolo coco weights and place them in folder yolo-coco

YOLO COCO WEIGHTS : https://pjreddie.com/media/files/yolov3.weights