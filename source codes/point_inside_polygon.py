def point_inside_polygon(point, polygon):
    """
    Checks if a point is inside a polygon.

    Args:
        point (tuple): (x, y) coordinates of the point.
        polygon (numpy.ndarray): Array of polygon vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if (y1 < y <= y2 or y2 < y <= y1) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside