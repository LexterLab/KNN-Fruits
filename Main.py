import math

fruits_data = [
    {"color": "red", "shape": "round", "size": "medium", "label": "apple"},
    {"color": "green", "shape": "round", "size": "small", "label": "grape"},
    {"color": "yellow", "shape": "oval", "size": "large", "label": "banana"},
    {"color": "orange", "shape": "round", "size": "large", "label": "orange"},
    {"color": "red", "shape": "round", "size": "large", "label": "watermelon"},
    {"color": "yellow", "shape": "oval", "size": "medium", "label": "lemon"},
    {"color": "purple", "shape": "round", "size": "medium", "label": "plum"},
    {"color": "green", "shape": "oval", "size": "medium", "label": "kiwi"},
    {"color": "yellow", "shape": "round", "size": "large", "label": "pineapple"},
    {"color": "orange", "shape": "oval", "size": "medium", "label": "peach"},
    {"color": "red", "shape": "oval", "size": "large", "label": "pomegranate"},
    {"color": "orange", "shape": "round", "size": "medium", "label": "apricot"},
    {"color": "green", "shape": "oval", "size": "medium", "label": "avocado"},
    {"color": "yellow", "shape": "round", "size": "small", "label": "pear"},
    {"color": "red", "shape": "round", "size": "large", "label": "cherry"},
    {"color": "purple", "shape": "round", "size": "medium", "label": "fig"},
    {"color": "green", "shape": "oval", "size": "large", "label": "honeydew melon"},
    {"color": "yellow", "shape": "oval", "size": "medium", "label": "mango"},
    {"color": "orange", "shape": "round", "size": "small", "label": "tangerine"},
    {"color": "red", "shape": "oval", "size": "medium", "label": "cranberry"}
]

color_to_num = {"red": 0, "orange": 1, "yellow": 2, "green": 3, "purple": 4}
shape_to_num = {"round": 0, "oval": 1}
size_to_num = {"small": 0, "medium": 1, "large": 2}


def calculate_accuracy(test_data, k, training_data):
    correct_predictions = 0
    total_predictions = len(test_data)
    for data_point in test_data:
        predicted_label = knn_classify(k, training_data, data_point)
        if predicted_label == data_point["label"]:
            correct_predictions += 1
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def euclidean_distance(point1, point2):
    distance = 0
    for key in point1:
        if key == "label":
            continue
        distance += (point1[key] - point2[key]) ** 2
    return math.sqrt(distance)


def knn_classify(k, training_data, new_data):
    distances = []
    for fruit in training_data:
        fruit_features = {
            "color": color_to_num[fruit["color"]],
            "shape": shape_to_num[fruit["shape"]],
            "size": size_to_num[fruit["size"]]
        }
        new_features = {
            "color": color_to_num[new_data["color"]],
            "shape": shape_to_num[new_data["shape"]],
            "size": size_to_num[new_data["size"]]
        }
        dist = euclidean_distance(fruit_features, new_features)
        distances.append((dist, fruit["label"]))
    distances.sort()
    nearest_neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in nearest_neighbors]

    label_counts = {label: labels.count(label) for label in set(labels)}

    return max(label_counts, key=label_counts.get)


new_fruit = {"color": "yellow", "shape": "oval", "size": "large"}
k = 3
predicted_label = knn_classify(k, fruits_data, new_fruit)
print("Predicted label:", predicted_label)

accuracy = calculate_accuracy(fruits_data, k, fruits_data)
print("Accuracy:", accuracy, "%")
