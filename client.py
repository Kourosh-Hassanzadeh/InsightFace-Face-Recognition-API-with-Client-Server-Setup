import base64
import requests


def load_pairs(pairs_file):
    try:
        pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                pair1 = lines[i].strip().split()
                pair2 = lines[i+1].strip().split()
                pairs.append((pair1[0], pair2[0], int(pair1[1])))
        return pairs
    except Exception as e:
        raise RuntimeError(f"Error loading pairs: {str(e)}")


def load_pairs_lfw(pairs_file):
    try:
        pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    folder = parts[0]
                    img1_idx = parts[1]
                    img2_idx = parts[2]
                    pairs.append((f"{folder}/{folder}_{img1_idx.zfill(4)}.jpg",
                                  f"{folder}/{folder}_{img2_idx.zfill(4)}.jpg", 1))
                elif len(parts) == 4:
                    folder1 = parts[0]
                    img1_idx = parts[1]
                    folder2 = parts[2]
                    img2_idx = parts[3]
                    pairs.append((f"{folder1}/{folder1}_{img1_idx.zfill(4)}.jpg",
                                  f"{folder2}/{folder2}_{img2_idx.zfill(4)}.jpg", 0))
        return pairs
    except Exception as e:
        raise RuntimeError(f"Error loading pairs: {str(e)}")


def load_image(file_path):
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Error loading image {file_path}: {str(e)}")


def evaluate_model(model, pairs, dataset_path):
    correct = 0
    total = len(pairs)
    fmr = 0
    fnmr = 0

    for i, pair in enumerate(pairs):
        img1_path = f"{dataset_path}/{pair[0]}"
        img2_path = f"{dataset_path}/{pair[1]}"
        label = int(pair[2])
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        if img1 is None or img2 is None:
            raise RuntimeError(f"Error loading images for pair {i}")

        try:
            response = requests.post(
                'http://127.0.0.1:5000/testModel', json={'img1': img1, 'img2': img2, 'model': model})

            if response.status_code == 200:
                res = response.json()
                check_match = res['match']

                if check_match and label != 0:
                    correct += 1
                elif not check_match and label == 0:
                    correct += 1
                elif check_match and label == 0:
                    fmr += 1
                elif not check_match and label != 0:
                    fnmr += 1
            else:
                raise RuntimeError(
                    f"Error in response: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed for pair {i}: {str(e)}")

    accuracy = correct / total
    return accuracy, fmr, fnmr


# set the path of pairs.txt here:
# lfw/pairs.txt | calfw/calfw/pairs_CALFW.txt | cplfw/cplfw/pairs_CPLFW.txt
pairs_file = 'calfw/calfw/pairs_CALFW.txt'

# if you use lfw dataset, use load_pairs_lfw() method
pairs = load_pairs(pairs_file)
# pairs = load_pairs_lfw(pairs_file)

# set the path of images folder:
# lfw/lfw | calfw/calfw/aligned images | cplfw/cplfw/aligned images
dataset_path = 'calfw/calfw/aligned images'

# set the model here: buffalo_l / bufallo_s
model = 'buffalo_s'
accuracy, fmr, fnmr = evaluate_model(model, pairs, dataset_path)

print(f"{model} - Accuracy: {accuracy}, FMR: {fmr}, FNMR: {fnmr}")
