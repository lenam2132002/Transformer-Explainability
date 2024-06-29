import requests
import os

data_dir = 'D:\\Bert_ex\\Transformer-Explainability\\data\\capec\\docs'
i = 0

for filename in os.listdir(data_dir):
    if filename.startswith("274_") and filename.endswith(".txt"):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as file:
            if(i > 20):
                break
            lines = file.readlines()
            for line in lines:
                # Tách dòng theo dấu cách
                parts = line.split()
                if parts and parts[0] != "GET":
                    if len(parts) > 1:
                        # print(parts[1])
                        # URL của server localhost
                        url = "http://127.0.0.1:80"  # Thay đổi theo endpoint của bạn

                        api = parts[1]
                        url = url + api
                        try:
                            # Gửi yêu cầu GET
                            response = requests.get(url)
                            i = i + 1
                            
                            # Kiểm tra mã trạng thái HTTP
                            if response.status_code == 200:
                                print("Yêu cầu thành công!")
                                print("Phản hồi từ server:", response.json())
                            else:
                                print(f"Lỗi: {response.status_code}")

                        except requests.exceptions.RequestException as e:
                            print(f"Lỗi khi gửi yêu cầu: {e}")
                    else:
                        print("Không có phần tử thứ hai trong dòng:", line)
print(f"so luong request: {i}")
