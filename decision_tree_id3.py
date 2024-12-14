from __future__ import print_function  # Đảm bảo tương thích với phiên bản Python 2 và 3
import numpy as np

class TreeNode(object):
    def __init__(self, ids=None, children=[], entropy=0, depth=0):
        self.ids = ids  # Danh sách các chỉ số dữ liệu trong nút này
        self.entropy = entropy  # Độ đo entropy của nút, sẽ được tính sau
        self.depth = depth  # Độ sâu của nút so với nút gốc
        self.split_attribute = None  # Thuộc tính được chọn để chia nhỏ, nếu không phải là nút lá
        self.children = children  # Danh sách các nút con của nút hiện tại
        self.order = None  # Thứ tự các giá trị của thuộc tính chia nhỏ trong các nút con
        self.label = None  # Nhãn của nút nếu nó là nút lá
        
    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute  
        self.order = order  

    def set_label(self, label):
        self.label = label 
        
def entropy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]  # Loại bỏ tần số bằng 0
    prob_0 = freq_0 / float(freq_0.sum())  # Tính xác suất của mỗi tần số
    return -np.sum(prob_0 * np.log(prob_0))  # Tính entropy

class DecisionTreeID3(object):
    def __init__(self, max_depth=8, min_samples_split=2, min_gain=1e-4, max_bin=8):
        self.root = None  # Nút gốc của cây quyết định
        self.max_depth = max_depth  # Độ sâu tối đa của cây
        self.min_samples_split = min_samples_split  # Số mẫu tối thiểu để chia nhỏ một nút
        self.Ntrain = 0  # Số lượng mẫu trong tập huấn luyện
        self.min_gain = min_gain  # Mức gain tối thiểu để chia nhỏ tiếp tục
        self.max_bin = max_bin # Số lượng ngưỡng tối đa để phân chia dữ liệu liên tục
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]  # Đếm số lượng mẫu trong tập dữ liệu
        self.data = data  
        self.attributes = list(data)  # Lấy danh sách các thuộc tính
        self.target = target  
        self.labels = target.unique()  # Lấy các nhãn mục tiêu duy nhất
        
        ids = range(self.Ntrain)  # Tạo danh sách các chỉ số mẫu dữ liệu
        self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)  # Tạo nút gốc với toàn bộ dữ liệu
        queue = [self.root]  
        while queue:
            node = queue.pop()  
            # Nếu chưa đạt tới độ sâu giới hạn hoặc entropy vẫn lớn hơn mức thông tin nhỏ nhất để chia
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)  
                if not node.children:  # Nếu không có nút con, đặt nhãn cho nút
                    self._set_label(node)
                queue += node.children  # Thêm các nút con vào hàng đợi
            else:
                self._set_label(node)  # Nếu không thể chia nhỏ thêm, đặt nhãn cho nút
                
    def _entropy(self, ids):
        if len(ids) == 0:
            return 0
        freq = np.array(self.target.iloc[ids].value_counts())  # Đếm tần số của các nhãn mục tiêu
        return entropy(freq)  # Tính entropy dựa trên tần số
    
    def _set_label(self, node):
        target_ids = node.ids
        node.set_label(self.target.iloc[target_ids].mode()[0])  # Đặt nhãn cho nút lá bằng giá trị mode của các nhãn mục tiêu
    
    def _split(self, node):
        ids = node.ids  # Lấy danh sách các chỉ số dữ liệu trong nút hiện tại
        best_gain = 0  
        best_splits = []  # Khởi tạo danh sách các cách chia nhỏ tốt nhất
        best_attribute = None 
        order = None  
        sub_data = self.data.iloc[ids, :]  # Lấy tập dữ liệu con tương ứng với các chỉ số dữ liệu của nút hiện tại

        for i, att in enumerate(self.attributes):  # Duyệt qua từng đặc trưng 
            values = sub_data[att].unique()  
            if len(values) == 1:  # Nếu đặc trưng chỉ có một giá trị duy nhất, bỏ qua đặc trưng này
                continue
            if np.issubdtype(values.dtype, np.number):  # Kiểm tra nếu thuộc tính là liên tục
                # Lấy các ngưỡng chia 
                sorted_values = np.sort(values)
                thresholds = [(sorted_values[j] + sorted_values[j + 1]) / 2 for j in range(len(sorted_values) - 1)]
                # Giới hạn ngưỡng chia
                if len(thresholds) > self.max_bin:
                    thresholds = np.random.choice(thresholds, self.max_bin, replace=False)
                    
                for threshold in thresholds:  # Duyệt qua các ngưỡng để kiểm tra
                    splits = [
                        [self.data.index.get_loc(idx) for idx in sub_data.index[sub_data[att] <= threshold]],  # Dữ liệu có giá trị <= ngưỡng
                        [self.data.index.get_loc(idx) for idx in sub_data.index[sub_data[att] > threshold]]  # Dữ liệu có giá trị > ngưỡng
                    ]
                    if min(map(len, splits)) < self.min_samples_split:  # Kiểm tra nếu một trong các cách chia nhỏ có số lượng mẫu nhỏ hơn min_samples_split
                        continue
                    HxS = 0  # Khởi tạo tổng entropy có trọng số
                    for split in splits:  
                        HxS += len(split) * self._entropy(split) / len(ids)  # Tính entropy có trọng số của cách chia nhỏ hiện tại
                    gain = node.entropy - HxS  # Tính gain cho cách chia nhỏ hiện tại
                    if gain < self.min_gain:  
                        continue
                    if gain > best_gain:  
                        best_gain = gain
                        best_splits = splits
                        best_attribute = att
                        order = [f"<= {threshold}", f"> {threshold}"]
            else:  # Xử lý các thuộc tính categorical
                splits = []  
                for val in values:  
                    sub_ids = sub_data.index[sub_data[att] == val].tolist()  # Lấy danh sách các chỉ số dữ liệu có giá trị hiện tại của đặc trưng
                    splits.append([self.data.index.get_loc(idx) for idx in sub_ids])  # Thêm danh sách chỉ số vào danh sách các cách chia nhỏ
                if min(map(len, splits)) < self.min_samples_split:  
                    continue
                HxS = 0  # Khởi tạo tổng entropy có trọng số
                for split in splits:  # Duyệt qua từng cách chia nhỏ
                    HxS += len(split) * self._entropy(split) / len(ids)  # Tính entropy có trọng số của cách chia nhỏ hiện tại
                gain = node.entropy - HxS  # Tính gain cho cách chia nhỏ hiện tại
                if gain < self.min_gain:  
                    continue
                if gain > best_gain:  
                    best_gain = gain
                    best_splits = splits
                    best_attribute = att
                    order = values
                    
        node.set_properties(best_attribute, order)  # Thiết lập thuộc tính chia nhỏ và thứ tự giá trị tốt nhất cho nút hiện tại
        child_nodes = [
            TreeNode(ids=split, entropy=self._entropy(split), depth=node.depth + 1)  
            for split in best_splits
        ]
        return child_nodes  # Trả về danh sách các nút con


    def predict(self, new_data):
        """
        :param new_data: a new dataframe, each row is a datapoint
        :return: predicted labels for each row
        """
        npoints = new_data.count()[0]
        labels = [None] * npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # Lấy một điểm dữ liệu 
            node = self.root
            while node.children:
                if isinstance(node.order[0], str) and "<=" in node.order[0]:  # Kiểm tra nếu thuộc tính là liên tục
                    threshold = float(node.order[0].split(" ")[1])  # Tách ngưỡng từ chuỗi
                    if x[node.split_attribute] <= threshold:
                        node = node.children[0]  # Đi xuống nhánh trái nếu giá trị nhỏ hơn hoặc bằng ngưỡng
                    else:
                        node = node.children[1]  # Đi xuống nhánh phải nếu giá trị lớn hơn ngưỡng
                else:
                    node = node.children[node.order.index(x[node.split_attribute])]  # Xử lý thuộc tính categorical
            labels[n] = node.label  
            
        return labels  
