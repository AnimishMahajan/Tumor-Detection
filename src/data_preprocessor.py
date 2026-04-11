from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoder.fit([[0], [1]])

#0 - Tumour
#1 - Normal

def image_to_numpy(folder_path, label):
    data = []
    paths = []
    result = []

    folder_path = Path(folder_path)
    for file in folder_path.glob('*.[jJ][pP][gG]'):
        paths.append(file)

    for path in paths[:-5]:
        img = Image.open(path).convert('RGB')
        img = img.resize((128,128))
        img = np.array(img)
        if(img.shape == (128,128,3)):
            data.append(np.array(img))
            result.append(encoder.transform([[label]]).toarray())
            
    # print(len(data))
    
    return np.array(data), np.array(result)

#Split data and save in processed folder files

data1, result1 = (image_to_numpy(r'data/raw/yes', 0))
data2, result2 = (image_to_numpy(r'data/raw/no', 1))

data = np.concatenate((data1, data2))
result = np.concatenate((result1, result2))

# data = data / 255.0
result = result.reshape(-1, 2)

print(f"Data shape: {data.shape}")
print(f"Result shape: {result.shape}")

x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, random_state=42)

np.save(r'data/processed/x_train.npy', x_train)
np.save(r'data/processed/x_test.npy', x_test)
np.save(r'data/processed/y_train.npy', y_train)
np.save(r'data/processed/y_test.npy', y_test)