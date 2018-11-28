import math


def facial_ratio(x1, y1, x2, y2):
    dist1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist1


# Generates features from coordinates of points
def generate_features(X, Y, shape):
    dist = []
    for i in range(0, int(shape[1] / 2) - 1):
        x1 = X[i]
        y1 = Y[i]
        for j in range(i + 1, int(shape[1] / 2)):
            x2 = X[j]
            y2 = Y[j]

            dist.append(facial_ratio(x1, y1, x2, y2))
    feature = []
    for i in range(0, len(dist)):
        dist1 = dist[i]
        for j in range(i + 1, len(dist)):
            dist2 = dist[j]

            feature.append(dist1 / dist2)

    return feature
