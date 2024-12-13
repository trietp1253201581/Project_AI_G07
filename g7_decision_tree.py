import numpy as np

# Class đại diện cho một nút trong cây quyết định
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Đặc trưng sử dụng để chia dữ liệu
        self.threshold = threshold  # Ngưỡng sử dụng để chia dữ liệu
        self.left = left            # Con trỏ tới nút con bên trái
        self.right = right          # Con trỏ tới nút con bên phải
        self.value = value          # Giá trị của nút nếu là nút lá

# Hàm tính chỉ số Gini để đo độ thuần nhất của nút
def gini(y):
    _, counts = np.unique(y, return_counts=True)  # Tìm các lớp và số lượng phần tử trong mỗi lớp
    gini = 1.0 - sum((count / len(y)) ** 2 for count in counts)  # Tính chỉ số Gini
    return gini

class G07DecisionTree():
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = TreeNode()

    # Hàm fit tree với datasets
    def fit(self, X, y):
        self.tree = self.build_tree_(X, y, depth=0, max_depth=self.max_depth)
    
    # Hàm dự đoán một tập dữ liệu
    def predict(self, X):
        return np.array([self.predict_tree_(self.tree, x) for x in X])
    
    # Hàm chia dữ liệu theo đặc trưng và ngưỡng
    def split_(self, X, y, feature, threshold):
        left_mask = X[:, feature] <= threshold  # Mặt nạ để lấy các phần tử nhỏ hơn hoặc bằng ngưỡng
        right_mask = X[:, feature] > threshold  # Mặt nạ để lấy các phần tử lớn hơn ngưỡng
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    # Hàm tìm đặc trưng và ngưỡng tốt nhất để chia dữ liệu
    def best_split_(self, X, y):
        best_gini = 1.0
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):  # Duyệt qua từng đặc trưng
            thresholds = np.unique(X[:, feature])  # Tìm tất cả các ngưỡng duy nhất
            for threshold in thresholds:  # Duyệt qua từng ngưỡng
                X_left, X_right, y_left, y_right = self.split_(X, y, feature, threshold)  # Chia dữ liệu
                if len(y_left) == 0 or len(y_right) == 0:  # Nếu một trong hai phần trống, bỏ qua
                    continue
                gini_left = gini(y_left)  # Tính chỉ số Gini cho phần bên trái
                gini_right = gini(y_right)  # Tính chỉ số Gini cho phần bên phải
                gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)  # Tính chỉ số Gini trung bình
                if gini_split < best_gini:  # Nếu Gini nhỏ hơn, cập nhật đặc trưng và ngưỡng tốt nhất
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    # Hàm xây dựng cây quyết định đệ quy
    def build_tree_(self, X, y, depth=0, max_depth=10):
        if len(np.unique(y)) == 1:  # Nếu tất cả các phần tử cùng một lớp, trả về nút lá
            return TreeNode(value=y[0])
        if depth >= max_depth:  # Nếu độ sâu đạt giới hạn, trả về nút lá
            return TreeNode(value=np.bincount(y).argmax())  # Trả về lớp phổ biến nhất
        feature, threshold = self.best_split_(X, y)  # Tìm đặc trưng và ngưỡng tốt nhất
        if feature is None:  # Nếu không tìm được đặc trưng tốt, trả về nút lá
            return TreeNode(value=np.bincount(y).argmax())  # Trả về lớp phổ biến nhất
        X_left, X_right, y_left, y_right = self.split_(X, y, feature, threshold)  # Chia dữ liệu
        left_child = self.build_tree_(X_left, y_left, depth + 1, max_depth)  # Xây dựng nút con bên trái
        right_child = self.build_tree_(X_right, y_right, depth + 1, max_depth)  # Xây dựng nút con bên phải
        return TreeNode(feature=feature, threshold=threshold, left=left_child, right=right_child)
    
    # Hàm dự đoán giá trị dựa trên cây quyết định
    def predict_tree_(self, node, X):
        if node.value is not None:  # Nếu là nút lá, trả về giá trị của nút lá
            return node.value
        if X[node.feature] <= node.threshold:  # Nếu giá trị nhỏ hơn hoặc bằng ngưỡng, duyệt cây con bên trái
            return self.predict_tree_(node.left, X)
        else:  # Nếu giá trị lớn hơn ngưỡng, duyệt cây con bên phải
            return self.predict_tree_(node.right, X)