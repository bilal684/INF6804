highway : python .\yolo.py --path data/highway/input --frames 1700 --output bilal --yolo yolo-coco --groundtruth data/highway/groundtruth --skip 470

canoe : python .\yolo.py --path data/canoe/input --frames 1189 --output canoe --yolo yolo-coco --groundtruth data/canoe/groundtruth --skip 800

streetCornerAtNight : python .\yolo.py --path data/streetCornerAtNight/input --frames 5200 --output streetCornerAtNight --yolo yolo-coco --groundtruth data/streetCornerAtNight/groundtruth --skip 800

turnpike : python .\yolo.py --path data/turnpike/input --frames 1500 --output turnpike --yolo yolo-coco --groundtruth data/turnpike/groundtruth --skip 800