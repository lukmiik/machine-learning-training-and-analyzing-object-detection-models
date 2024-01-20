import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(confusion_matrix: np.ndarray) -> np.float64:
    return round(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix), 2)


def calculate_recall_for_each_class(confusion_matrix: np.ndarray):
    num_classes = len(confusion_matrix) - 1
    recalls = []
    for i in range(num_classes):
        recall = confusion_matrix[i, i] / (
            confusion_matrix[i, i] + confusion_matrix[i, -1]
        )
        recalls.append(round(recall, 2))

    return recalls


def create_and_save_accuracies_plot(accuracies: list) -> None:
    plt.figure()
    plt.bar(MODELS, accuracies, color=COLORS)
    plt.ylim(0, 1)
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.savefig('model_accuracies.png')


def create_and_save_reccalls_plot(recalls: list) -> None:
    bar_width = 0.2
    index = np.arange(len(CLASSES))

    plt.figure()
    plt.bar(
        index - bar_width,
        recalls[0],
        width=bar_width,
        label=MODELS[0],
        color=COLORS[0],
    )
    plt.bar(index, recalls[1], width=bar_width, label=MODELS[1], color=COLORS[1])
    plt.bar(
        index + bar_width,
        recalls[2],
        width=bar_width,
        label=MODELS[2],
        color=COLORS[2],
    )
    plt.xlabel('Classes')
    plt.ylabel('Recall')
    plt.title('Recall Comparison for Each Class and Model')
    plt.xticks(index, CLASSES)
    plt.legend()
    plt.savefig('model_recalls.png')


MODELS = ['YOLOv8', 'YOLO-NAS', 'YOLOv5']
COLORS = ['blue', 'red', 'green']
CLASSES = [
    'cat',
    'chicken',
    'cow',
    'dog',
    'fox',
    'goat',
    'horse',
    'person',
    'racoon',
    'skunk',
]

yolov8_confusion_matrix = np.array(
    [
        [5, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 11, 0, 0, 1, 0, 0, 0, 0, 0, 3],
        [0, 0, 13, 1, 0, 2, 1, 0, 0, 0, 4],
        [0, 0, 0, 13, 1, 1, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 19, 0, 0, 0, 1, 8],
        [0, 0, 0, 1, 0, 0, 6, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0],
        [0, 2, 3, 0, 1, 4, 2, 5, 3, 1, 0],
    ]
)
yolo_nas_confusion_matrix = np.array(
    [
        [2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 1],
        [1, 7, 0, 0, 0, 0, 1, 0, 0, 0, 6],
        [0, 0, 3, 0, 0, 8, 5, 0, 0, 0, 5],
        [0, 0, 0, 5, 0, 1, 1, 4, 0, 0, 5],
        [0, 0, 0, 0, 3, 0, 2, 0, 0, 1, 3],
        [0, 0, 1, 1, 0, 15, 2, 0, 0, 0, 10],
        [0, 0, 0, 1, 0, 0, 6, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2, 12, 0, 0, 11],
        [1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0],
        [0, 2, 0, 0, 0, 3, 5, 9, 1, 1, 0],
    ]
)
yolov5_confusion_matrix = np.array(
    [
        [2, 0, 0, 1, 0, 0, 0, 2, 0, 1, 1],
        [1, 7, 0, 0, 0, 0, 1, 0, 0, 0, 6],
        [0, 0, 3, 0, 0, 8, 5, 0, 0, 0, 5],
        [0, 0, 0, 6, 0, 1, 1, 4, 0, 0, 5],
        [0, 0, 0, 0, 3, 0, 2, 0, 0, 1, 3],
        [0, 0, 1, 1, 0, 15, 2, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0, 2, 12, 0, 0, 11],
        [1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0],
        [0, 2, 0, 0, 0, 3, 5, 9, 1, 1, 0],
    ]
)

yolov8_accuracy = calculate_accuracy(yolov8_confusion_matrix)
yolo_nas_accuracy = calculate_accuracy(yolo_nas_confusion_matrix)
yolov5_accuracy = calculate_accuracy(yolov5_confusion_matrix)
accuracies = [yolov8_accuracy, yolo_nas_accuracy, yolov5_accuracy]

yolov8_recalls = calculate_recall_for_each_class(yolov8_confusion_matrix)
yolo_nas_recalls = calculate_recall_for_each_class(yolo_nas_confusion_matrix)
yolov5_recalls = calculate_recall_for_each_class(yolov5_confusion_matrix)
recalls = [yolov8_recalls, yolo_nas_recalls, yolov5_recalls]

create_and_save_accuracies_plot(accuracies)
create_and_save_reccalls_plot(recalls)
